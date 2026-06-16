//! Loss Calculations
//!
use candle::backend::BackendStorage;
use candle::{CpuStorage, DType, Layout, Result, Shape, Tensor, WithDType};
use rayon::prelude::*;

/// The negative log likelihood loss.
///
/// Arguments
///
/// * [inp]: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
///   of categories. This is expected to contain log probabilities.
/// * [target]: The ground truth labels as a tensor of u32 of dimension `N`.
///
/// The resulting tensor is a scalar containing the average value over the batch.
pub fn nll(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    let b_sz = match target.dims() {
        &[b_sz] => b_sz,
        dims => candle::bail!("the target tensor should have a single dimension ({dims:?})"),
    };
    match inp.dims() {
        &[inp_b_sz, _] => {
            if inp_b_sz != b_sz {
                candle::bail!("batch size mismatch between inp ({inp_b_sz}) and target ({b_sz})")
            }
        }
        dims => candle::bail!("the target tensor should have two dimensions ({dims:?})"),
    }
    inp.gather(&target.unsqueeze(1)?, 1)?
        .sum_all()?
        .affine(-1f64 / b_sz as f64, 0.)
}

/// The cross-entropy loss.
///
/// Arguments
///
/// * [inp]: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
///   of categories. This is expected to raw logits.
/// * [target]: The ground truth labels as a tensor of u32 of dimension `N`.
///
/// The resulting tensor is a scalar containing the average value over the batch.
pub fn cross_entropy(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    if inp.rank() != 2 {
        candle::bail!("cross_entropy expects an input tensor of rank 2")
    }
    cross_entropy_dims(inp.layout(), target.layout())?;
    validate_cross_entropy_target_dtype(target)?;
    let inp = inp.contiguous()?;
    let target = target.contiguous()?;
    // CPU `target_indices` handles U8/U32/I64 directly with per-element
    // validation, so keep the original dtype for precise errors. GPU kernels
    // consume U32 targets; only U8 is cast because it is lossless. Casting I64
    // on device can wrap invalid labels into valid class ids before the kernel
    // can reject them, so reject it explicitly instead of synchronizing for a
    // host-side validation pass.
    let target = match (target.device().is_cpu(), target.dtype()) {
        (true, _) | (false, DType::U32) => target,
        (false, DType::U8) => target.to_dtype(DType::U32)?,
        (false, DType::I64) => candle::bail!(
            "cross_entropy target dtype I64 is not supported on CUDA/Metal; use U32 or U8 targets"
        ),
        (false, dtype) => candle::bail!("unsupported cross_entropy target dtype {dtype:?}"),
    };
    inp.apply_op2(&target, CrossEntropyRows)?.mean_all()
}

struct CrossEntropyRows;

fn cross_entropy_dims(inp: &Layout, target: &Layout) -> Result<(usize, usize)> {
    let b_sz = match target.dims() {
        &[b_sz] => b_sz,
        dims => candle::bail!("the target tensor should have a single dimension ({dims:?})"),
    };
    match inp.dims() {
        &[inp_b_sz, n_classes] => {
            if inp_b_sz != b_sz {
                candle::bail!("batch size mismatch between inp ({inp_b_sz}) and target ({b_sz})")
            }
            if n_classes == 0 {
                candle::bail!("cross_entropy expects at least one class")
            }
            Ok((inp_b_sz, n_classes))
        }
        dims => candle::bail!("the input tensor should have two dimensions ({dims:?})"),
    }
}

fn check_target_index(idx: usize, n_classes: usize) -> Result<usize> {
    if idx >= n_classes {
        candle::bail!("target index {idx} is out of bounds for {n_classes} classes")
    }
    Ok(idx)
}

fn check_target_i64_index(idx: i64, n_classes: usize) -> Result<usize> {
    if idx < 0 {
        candle::bail!("target index {idx} is negative")
    }
    let idx = usize::try_from(idx)
        .map_err(|_| candle::Error::msg(format!("target index {idx} cannot be cast to usize")))?;
    check_target_index(idx, n_classes)
}

fn validate_cross_entropy_target_dtype(target: &Tensor) -> Result<()> {
    match target.dtype() {
        DType::U8 | DType::U32 | DType::I64 => Ok(()),
        dtype => candle::bail!("cross_entropy target dtype must be U8, U32, or I64, got {dtype:?}"),
    }
}

fn target_indices(
    storage: &CpuStorage,
    layout: &Layout,
    b_sz: usize,
    n_classes: usize,
) -> Result<Vec<usize>> {
    let (o1, o2) = match layout.contiguous_offsets() {
        None => candle::bail!("target has to be contiguous"),
        Some(o) => o,
    };
    if o2 - o1 != b_sz {
        candle::bail!(
            "target has an unexpected number of elements, expected {b_sz}, got {}",
            o2 - o1
        )
    }

    match storage {
        CpuStorage::U32(target) => target[o1..o2]
            .iter()
            .map(|&idx| check_target_index(idx as usize, n_classes))
            .collect(),
        CpuStorage::U8(target) => target[o1..o2]
            .iter()
            .map(|&idx| check_target_index(idx as usize, n_classes))
            .collect(),
        CpuStorage::I64(target) => target[o1..o2]
            .iter()
            .map(|&idx| check_target_i64_index(idx, n_classes))
            .collect(),
        storage => candle::bail!(
            "unsupported target dtype for cross_entropy {:?}",
            storage.dtype()
        ),
    }
}

// CPU forward for the fused row losses. Reductions accumulate in `f64`
// regardless of the input dtype (BF16/F16/F32/F64). This is conservative —
// it costs the same as `f32` accumulation on modern CPUs while keeping the
// log-sum-exp numerically stable for large `n_classes`. The output is cast
// back to `T`, matching the input dtype.
fn cross_entropy_rows_cpu_fwd<T: WithDType>(
    inp: &[T],
    inp_l: &Layout,
    target: &[usize],
) -> Result<(CpuStorage, Shape)> {
    let (b_sz, n_classes) = match inp_l.dims() {
        &[b_sz, n_classes] => (b_sz, n_classes),
        dims => candle::bail!("the input tensor should have two dimensions ({dims:?})"),
    };
    let (o1, o2) = match inp_l.contiguous_offsets() {
        None => candle::bail!("input has to be contiguous"),
        Some(o) => o,
    };
    let inp = &inp[o1..o2];
    let mut dst = vec![T::from_f64(0.); b_sz];
    inp.par_chunks(n_classes)
        .zip(dst.par_iter_mut())
        .zip(target.par_iter())
        .for_each(|((inp, dst), &target)| {
            let max = inp
                .iter()
                .map(|v| v.to_f64())
                .fold(f64::NEG_INFINITY, f64::max);
            let sum_exp = inp.iter().map(|v| (v.to_f64() - max).exp()).sum::<f64>();
            *dst = T::from_f64(max + sum_exp.ln() - inp[target].to_f64());
        });
    Ok((T::to_cpu_storage_owned(dst), Shape::from_dims(&[b_sz])))
}

impl candle::CustomOp2 for CrossEntropyRows {
    fn name(&self) -> &'static str {
        "cross-entropy-rows"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (b_sz, n_classes) = cross_entropy_dims(l1, l2)?;
        let target = target_indices(s2, l2, b_sz, n_classes)?;
        match s1 {
            CpuStorage::BF16(s1) => cross_entropy_rows_cpu_fwd::<half::bf16>(s1, l1, &target),
            CpuStorage::F16(s1) => cross_entropy_rows_cpu_fwd::<half::f16>(s1, l1, &target),
            CpuStorage::F32(s1) => cross_entropy_rows_cpu_fwd::<f32>(s1, l1, &target),
            CpuStorage::F64(s1) => cross_entropy_rows_cpu_fwd::<f64>(s1, l1, &target),
            storage => candle::bail!(
                "unsupported input dtype for cross_entropy {:?}",
                storage.dtype()
            ),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{LaunchAsync, LaunchConfig};
        use candle::cuda_backend::CudaStorageSlice;
        use candle::cuda_backend::{kernels, WrapErr};
        use candle::CudaStorage;

        let (b_sz, n_classes) = cross_entropy_dims(l1, l2)?;
        if !(l1.is_contiguous() && l2.is_contiguous()) {
            candle::bail!("Non contiguous cross_entropy is not implemented for CUDA");
        }

        let dev = s1.device();
        let target = match &s2.slice {
            CudaStorageSlice::U32(target) => target.slice(l2.start_offset()..),
            _ => candle::bail!("CUDA cross_entropy target must be U32"),
        };

        let block_size = n_classes.min(1024).next_power_of_two().max(32);
        let cfg = LaunchConfig {
            grid_dim: (b_sz as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let el = b_sz * n_classes;
        let dst = match &s1.slice {
            CudaStorageSlice::BF16(src) => {
                let func = dev.get_or_load_func("cross_entropy_fwd_bf16", &kernels::REDUCE)?;
                let dst = unsafe { dev.alloc::<half::bf16>(b_sz)? };
                let mut builder = func.builder();
                builder.arg(&el);
                builder.arg(&n_classes);
                builder.arg(&src.slice(l1.start_offset()..));
                builder.arg(&target);
                builder.arg(&dst);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::BF16(dst)
            }
            CudaStorageSlice::F16(src) => {
                let func = dev.get_or_load_func("cross_entropy_fwd_f16", &kernels::REDUCE)?;
                let dst = unsafe { dev.alloc::<half::f16>(b_sz)? };
                let mut builder = func.builder();
                builder.arg(&el);
                builder.arg(&n_classes);
                builder.arg(&src.slice(l1.start_offset()..));
                builder.arg(&target);
                builder.arg(&dst);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F16(dst)
            }
            CudaStorageSlice::F32(src) => {
                let func = dev.get_or_load_func("cross_entropy_fwd_f32", &kernels::REDUCE)?;
                let dst = unsafe { dev.alloc::<f32>(b_sz)? };
                let mut builder = func.builder();
                builder.arg(&el);
                builder.arg(&n_classes);
                builder.arg(&src.slice(l1.start_offset()..));
                builder.arg(&target);
                builder.arg(&dst);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F32(dst)
            }
            CudaStorageSlice::F64(src) => {
                let func = dev.get_or_load_func("cross_entropy_fwd_f64", &kernels::REDUCE)?;
                let dst = unsafe { dev.alloc::<f64>(b_sz)? };
                let mut builder = func.builder();
                builder.arg(&el);
                builder.arg(&n_classes);
                builder.arg(&src.slice(l1.start_offset()..));
                builder.arg(&target);
                builder.arg(&dst);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F64(dst)
            }
            s => candle::bail!("unsupported dtype for cross_entropy {:?}", s.dtype()),
        };
        Ok((
            CudaStorage {
                slice: dst,
                device: dev.clone(),
            },
            Shape::from_dims(&[b_sz]),
        ))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle::MetalStorage,
        l1: &Layout,
        s2: &candle::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        let (b_sz, n_classes) = cross_entropy_dims(l1, l2)?;
        if !(l1.is_contiguous() && l2.is_contiguous()) {
            candle::bail!("Non contiguous cross_entropy is not implemented for Metal");
        }
        let name = match (s1.dtype(), s2.dtype()) {
            (DType::F32, DType::U32) => "cross_entropy_fwd_f32",
            (DType::F16, DType::U32) => "cross_entropy_fwd_f16",
            (DType::BF16, DType::U32) => "cross_entropy_fwd_bf16",
            (dt1, dt2) => candle::bail!("cross_entropy is not implemented for {dt1:?} {dt2:?}"),
        };

        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("cross-entropy");
        let output = device.new_buffer(b_sz, s1.dtype(), "cross-entropy")?;
        candle_metal_kernels::call_cross_entropy_forward(
            device.metal_device(),
            &encoder,
            device.kernels(),
            name,
            b_sz * n_classes,
            n_classes,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let new_storage = candle::MetalStorage::new(output, device.clone(), b_sz, s1.dtype());
        Ok((new_storage, Shape::from_dims(&[b_sz])))
    }

    fn bwd(
        &self,
        arg1: &Tensor,
        arg2: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        // arg1 (logits) and arg2 (target) are already contiguous: the public
        // `cross_entropy` makes them contiguous before calling apply_op2.
        // grad_res may be a broadcast view of the scalar mean_all gradient,
        // so we still materialize it.
        let grad_res = grad_res.contiguous()?;
        let grad = arg1.apply_op3_no_bwd(arg2, &grad_res, &CrossEntropyRowsBackward)?;
        Ok((Some(grad), None))
    }
}

struct CrossEntropyRowsBackward;

// CPU backward. Same `f64` accumulation choice as the forward; the
// per-class softmax denominator is recomputed from the input rather than
// cached, so this op does not need a saved-tensor side channel.
fn cross_entropy_rows_cpu_bwd<T: WithDType>(
    inp: &[T],
    inp_l: &Layout,
    target: &[usize],
    grad_res: &[T],
    grad_res_l: &Layout,
) -> Result<(CpuStorage, Shape)> {
    let (b_sz, n_classes) = match inp_l.dims() {
        &[b_sz, n_classes] => (b_sz, n_classes),
        dims => candle::bail!("the input tensor should have two dimensions ({dims:?})"),
    };
    match grad_res_l.dims() {
        &[grad_b_sz] if grad_b_sz == b_sz => {}
        dims => candle::bail!("grad_res should have shape [{b_sz}], got {:?}", dims),
    }
    let (inp_o1, inp_o2) = match inp_l.contiguous_offsets() {
        None => candle::bail!("input has to be contiguous"),
        Some(o) => o,
    };
    let (grad_o1, grad_o2) = match grad_res_l.contiguous_offsets() {
        None => candle::bail!("grad_res has to be contiguous"),
        Some(o) => o,
    };
    let inp = &inp[inp_o1..inp_o2];
    let grad_res = &grad_res[grad_o1..grad_o2];
    let mut dst = vec![T::from_f64(0.); b_sz * n_classes];
    inp.par_chunks(n_classes)
        .zip(dst.par_chunks_mut(n_classes))
        .zip(target.par_iter())
        .zip(grad_res.par_iter())
        .for_each(|(((inp, dst), &target), &grad_res)| {
            let max = inp
                .iter()
                .map(|v| v.to_f64())
                .fold(f64::NEG_INFINITY, f64::max);
            let sum_exp = inp.iter().map(|v| (v.to_f64() - max).exp()).sum::<f64>();
            let grad_res = grad_res.to_f64();
            for (class_idx, (inp, dst)) in inp.iter().zip(dst.iter_mut()).enumerate() {
                let mut grad = (inp.to_f64() - max).exp() / sum_exp;
                if class_idx == target {
                    grad -= 1.;
                }
                *dst = T::from_f64(grad * grad_res);
            }
        });
    Ok((T::to_cpu_storage_owned(dst), inp_l.shape().clone()))
}

impl candle::CustomOp3 for CrossEntropyRowsBackward {
    fn name(&self) -> &'static str {
        "cross-entropy-rows-bwd"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (b_sz, n_classes) = cross_entropy_dims(l1, l2)?;
        let target = target_indices(s2, l2, b_sz, n_classes)?;
        match (s1, s3) {
            (CpuStorage::BF16(s1), CpuStorage::BF16(s3)) => {
                cross_entropy_rows_cpu_bwd::<half::bf16>(s1, l1, &target, s3, l3)
            }
            (CpuStorage::F16(s1), CpuStorage::F16(s3)) => {
                cross_entropy_rows_cpu_bwd::<half::f16>(s1, l1, &target, s3, l3)
            }
            (CpuStorage::F32(s1), CpuStorage::F32(s3)) => {
                cross_entropy_rows_cpu_bwd::<f32>(s1, l1, &target, s3, l3)
            }
            (CpuStorage::F64(s1), CpuStorage::F64(s3)) => {
                cross_entropy_rows_cpu_bwd::<f64>(s1, l1, &target, s3, l3)
            }
            (s1, s3) => {
                candle::bail!(
                    "unsupported dtype for cross_entropy backward {:?} {:?}",
                    s1.dtype(),
                    s3.dtype()
                )
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
        s3: &candle::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{LaunchAsync, LaunchConfig};
        use candle::cuda_backend::CudaStorageSlice;
        use candle::cuda_backend::{kernels, WrapErr};
        use candle::CudaStorage;

        let (b_sz, n_classes) = cross_entropy_dims(l1, l2)?;
        match l3.dims() {
            &[grad_b_sz] if grad_b_sz == b_sz => {}
            dims => candle::bail!("grad_res should have shape [{b_sz}], got {:?}", dims),
        }
        if !(l1.is_contiguous() && l2.is_contiguous() && l3.is_contiguous()) {
            candle::bail!("Non contiguous cross_entropy backward is not implemented for CUDA");
        }

        let dev = s1.device();
        let target = match &s2.slice {
            CudaStorageSlice::U32(target) => target.slice(l2.start_offset()..),
            _ => candle::bail!("CUDA cross_entropy target must be U32"),
        };

        let block_size = n_classes.min(1024).next_power_of_two().max(32);
        let cfg = LaunchConfig {
            grid_dim: (b_sz as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let el = b_sz * n_classes;
        let dst = match (&s1.slice, &s3.slice) {
            (CudaStorageSlice::BF16(src), CudaStorageSlice::BF16(grad_res)) => {
                let func = dev.get_or_load_func("cross_entropy_bwd_bf16", &kernels::REDUCE)?;
                let dst = unsafe { dev.alloc::<half::bf16>(el)? };
                let mut builder = func.builder();
                builder.arg(&el);
                builder.arg(&n_classes);
                builder.arg(&src.slice(l1.start_offset()..));
                builder.arg(&target);
                builder.arg(&grad_res.slice(l3.start_offset()..));
                builder.arg(&dst);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::BF16(dst)
            }
            (CudaStorageSlice::F16(src), CudaStorageSlice::F16(grad_res)) => {
                let func = dev.get_or_load_func("cross_entropy_bwd_f16", &kernels::REDUCE)?;
                let dst = unsafe { dev.alloc::<half::f16>(el)? };
                let mut builder = func.builder();
                builder.arg(&el);
                builder.arg(&n_classes);
                builder.arg(&src.slice(l1.start_offset()..));
                builder.arg(&target);
                builder.arg(&grad_res.slice(l3.start_offset()..));
                builder.arg(&dst);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F16(dst)
            }
            (CudaStorageSlice::F32(src), CudaStorageSlice::F32(grad_res)) => {
                let func = dev.get_or_load_func("cross_entropy_bwd_f32", &kernels::REDUCE)?;
                let dst = unsafe { dev.alloc::<f32>(el)? };
                let mut builder = func.builder();
                builder.arg(&el);
                builder.arg(&n_classes);
                builder.arg(&src.slice(l1.start_offset()..));
                builder.arg(&target);
                builder.arg(&grad_res.slice(l3.start_offset()..));
                builder.arg(&dst);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F32(dst)
            }
            (CudaStorageSlice::F64(src), CudaStorageSlice::F64(grad_res)) => {
                let func = dev.get_or_load_func("cross_entropy_bwd_f64", &kernels::REDUCE)?;
                let dst = unsafe { dev.alloc::<f64>(el)? };
                let mut builder = func.builder();
                builder.arg(&el);
                builder.arg(&n_classes);
                builder.arg(&src.slice(l1.start_offset()..));
                builder.arg(&target);
                builder.arg(&grad_res.slice(l3.start_offset()..));
                builder.arg(&dst);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F64(dst)
            }
            (s1, s3) => candle::bail!(
                "unsupported dtype for cross_entropy bwd {:?} {:?}",
                s1.dtype(),
                s3.dtype()
            ),
        };
        Ok((
            CudaStorage {
                slice: dst,
                device: dev.clone(),
            },
            l1.shape().clone(),
        ))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle::MetalStorage,
        l1: &Layout,
        s2: &candle::MetalStorage,
        l2: &Layout,
        s3: &candle::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        let (b_sz, n_classes) = cross_entropy_dims(l1, l2)?;
        match l3.dims() {
            &[grad_b_sz] if grad_b_sz == b_sz => {}
            dims => candle::bail!("grad_res should have shape [{b_sz}], got {:?}", dims),
        }
        if !(l1.is_contiguous() && l2.is_contiguous() && l3.is_contiguous()) {
            candle::bail!("Non contiguous cross_entropy backward is not implemented for Metal");
        }
        let name = match (s1.dtype(), s2.dtype(), s3.dtype()) {
            (DType::F32, DType::U32, DType::F32) => "cross_entropy_bwd_f32",
            (DType::F16, DType::U32, DType::F16) => "cross_entropy_bwd_f16",
            (DType::BF16, DType::U32, DType::BF16) => "cross_entropy_bwd_bf16",
            (dt1, dt2, dt3) => {
                candle::bail!(
                    "cross_entropy backward is not implemented for {dt1:?} {dt2:?} {dt3:?}"
                )
            }
        };

        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("cross-entropy-bwd");
        let elem_count = b_sz * n_classes;
        let output = device.new_buffer(elem_count, s1.dtype(), "cross-entropy-bwd")?;
        candle_metal_kernels::call_cross_entropy_backward(
            device.metal_device(),
            &encoder,
            device.kernels(),
            name,
            elem_count,
            n_classes,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            s3.buffer(),
            l3.start_offset() * s3.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let new_storage = candle::MetalStorage::new(output, device.clone(), elem_count, s1.dtype());
        Ok((new_storage, l1.shape().clone()))
    }
}

/// The mean squared error loss.
pub fn mse(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    (inp - target)?.sqr()?.mean_all()
}

/// The binary cross-entropy with logit loss.
///
/// Arguments
///
/// * [inp]: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
///   of categories. This is expected to raw logits.
/// * [target]: The ground truth labels as a tensor of u32 of dimension `N, C` where `N` is the batch size and `C` the number
///   of categories.
///
/// The resulting tensor is a scalar containing the average value over the batch.
pub fn binary_cross_entropy_with_logit(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    let inp = crate::ops::sigmoid(inp)?;

    let left_side = target * inp.log()?;
    let right_side = (target.affine(-1., 1.))? * inp.affine(-1., 1.)?.log()?;

    let loss = left_side? + right_side?;
    let loss = loss?.neg()?.mean_all()?;

    Ok(loss)
}

/// HuberLoss
///
/// A robust loss function that combines `MAE` and `MSE` losses:
///
/// - When the absolute element-wise error is less than `delta`, it uses a squared term (MSE loss).
/// - When the absolute element-wise error is greater than or equal to `delta`, it uses a linear term (MAE loss scaled by `delta`).
/// # Formula
///
/// HuberLoss =
/// ```tex
/// 0.5(x_n - y_n)^2, & |x_n - y_n| < delta
/// delta(|x_n - y_n| - 0.5delta), & |x_n - y_n| >= delta
/// ```
pub fn huber(inp: &Tensor, target: &Tensor, delta: f64) -> Result<Tensor> {
    if inp.dims() != target.dims() {
        candle::bail!(
            "input and target must have the same shape, got inp: {:?}, target: {:?}",
            inp.dims(),
            target.dims()
        );
    }
    let diff = (inp - target)?;
    let abs_diff = diff.abs()?;
    let mask = abs_diff.le(delta)?;
    let squared_loss = ((&diff * &diff)? * 0.5)?;
    let linear_loss = ((abs_diff * delta)? - 0.5 * delta.powi(2))?;
    let loss = mask.where_cond(&squared_loss, &linear_loss)?;
    loss.mean_all()
}
