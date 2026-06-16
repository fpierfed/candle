#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::test_utils::{to_vec0_round, to_vec2_round};
use candle::{Device, Result, Tensor, Var};
/* Equivalent python code:
import torch
import torch.nn.functional as F
input = torch.tensor([
    [ 1.1050,  0.3013, -1.5394, -2.1528, -0.8634],
    [ 1.0730, -0.9419, -0.1670, -0.6582,  0.5061],
    [ 0.8318,  1.1154, -0.3610,  0.5351,  1.0830]])

target = torch.tensor([1, 0, 4])
print(F.nll_loss(F.log_softmax(input, dim=1), target))
print(F.cross_entropy(input, target))
*/
#[test]
fn nll_and_cross_entropy() -> Result<()> {
    let cpu = Device::Cpu;
    let input = Tensor::new(
        &[
            [1.1050f32, 0.3013, -1.5394, -2.1528, -0.8634],
            [1.0730, -0.9419, -0.1670, -0.6582, 0.5061],
            [0.8318, 1.1154, -0.3610, 0.5351, 1.0830],
        ],
        &cpu,
    )?;
    let target = Tensor::new(&[1u32, 0, 4], &cpu)?;

    let log_softmax = candle_nn::ops::log_softmax(&input, 1)?;
    let loss = candle_nn::loss::nll(&log_softmax, &target)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 1.1312);
    let loss = candle_nn::loss::cross_entropy(&input, &target)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 1.1312);
    Ok(())
}

#[test]
fn cross_entropy_backward() -> Result<()> {
    let cpu = Device::Cpu;
    let input = Var::new(&[[1f32, 2., 0.], [0., -1., 1.]], &cpu)?;
    let target = Tensor::new(&[1u32, 2], &cpu)?;

    let loss = candle_nn::loss::cross_entropy(input.as_tensor(), &target)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 0.4076);

    let grads = loss.backward()?;
    let grad = grads.get(input.as_tensor()).unwrap();
    assert_eq!(
        to_vec2_round(grad, 4)?,
        &[[0.1224, -0.1674, 0.045], [0.1224, 0.045, -0.1674]]
    );
    Ok(())
}

fn assert_cross_entropy_err(input: &Tensor, target: &Tensor, expected: &str) {
    let err = candle_nn::loss::cross_entropy(input, target)
        .expect_err("cross_entropy should reject invalid targets");
    let msg = err.to_string();
    assert!(
        msg.contains(expected),
        "expected error containing {expected:?}, got {msg:?}"
    );
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn assert_close(actual: f32, expected: f32, tol: f32, context: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{context}: expected {expected}, got {actual}"
    );
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn assert_vec_close(actual: &[f32], expected: &[f32], tol: f32, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: length mismatch, expected {}, got {}",
        expected.len(),
        actual.len()
    );
    for (idx, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tol,
            "{context}: value mismatch at {idx}, expected {expected}, got {actual}"
        );
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn cross_entropy_device_u8_target(device: &Device) -> Result<()> {
    let input = Tensor::new(&[[1f32, 2., 0.], [0., -1., 1.]], device)?;
    let target = Tensor::new(&[1u8, 2], device)?;
    let loss = candle_nn::loss::cross_entropy(&input, &target)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 0.4076);
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn cross_entropy_device_rejects_i64_target(device: &Device) -> Result<()> {
    let input = Tensor::new(&[[1f32, 2., 0.], [0., -1., 1.]], device)?;
    let target = Tensor::new(&[u32::MAX as i64 + 1, 1], device)?;
    assert_cross_entropy_err(&input, &target, "I64 is not supported");
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn cross_entropy_device_invalid_u32_target_yields_nan(device: &Device) -> Result<()> {
    let input = Tensor::new(&[[1f32, 2., 0.], [0., -1., 1.]], device)?;
    let target = Tensor::new(&[1u32, 3], device)?;
    let loss = candle_nn::loss::cross_entropy(&input, &target)?;
    assert!(
        loss.to_scalar::<f32>()?.is_nan(),
        "expected NaN loss for out-of-range target on accelerator"
    );
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn cross_entropy_device_matches_cpu(device: &Device, n_classes: usize) -> Result<()> {
    let cpu = Device::Cpu;
    let b_sz = 3;
    let logits = (0..b_sz * n_classes)
        .map(|idx| {
            let row = idx / n_classes;
            let col = idx % n_classes;
            let wave = ((idx * 17 + row * 11) % 23) as f32;
            (wave - 11.) * 0.13 + col as f32 * 0.001 + row as f32 * 0.07
        })
        .collect::<Vec<_>>();
    let targets = vec![0u32, (n_classes / 2) as u32, (n_classes - 1) as u32];

    let cpu_input = Var::from_vec(logits.clone(), (b_sz, n_classes), &cpu)?;
    let cpu_target = Tensor::from_vec(targets.clone(), b_sz, &cpu)?;
    let cpu_loss_tensor = candle_nn::loss::cross_entropy(cpu_input.as_tensor(), &cpu_target)?;
    let cpu_loss = cpu_loss_tensor.to_scalar::<f32>()?;
    let cpu_grads = cpu_loss_tensor.backward()?;
    let cpu_grad = cpu_grads
        .get(cpu_input.as_tensor())
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let device_input = Var::from_vec(logits, (b_sz, n_classes), device)?;
    let device_target = Tensor::from_vec(targets, b_sz, device)?;
    let device_loss = candle_nn::loss::cross_entropy(device_input.as_tensor(), &device_target)?;
    let device_loss_scalar = device_loss.to_scalar::<f32>()?;
    let loss_context = format!("loss n_classes={n_classes}");
    assert_close(device_loss_scalar, cpu_loss, 5e-4, &loss_context);

    let device_grads = device_loss.backward()?;
    let device_grad = device_grads
        .get(device_input.as_tensor())
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let grad_context = format!("grad n_classes={n_classes}");
    assert_vec_close(&device_grad, &cpu_grad, 5e-4, &grad_context);
    Ok(())
}

#[test]
fn cross_entropy_target_dtypes() -> Result<()> {
    let cpu = Device::Cpu;
    let input = Tensor::new(&[[1f32, 2., 0.], [0., -1., 1.]], &cpu)?;

    let target = Tensor::new(&[1u8, 2], &cpu)?;
    let loss = candle_nn::loss::cross_entropy(&input, &target)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 0.4076);

    let target = Tensor::new(&[1i64, 2], &cpu)?;
    let loss = candle_nn::loss::cross_entropy(&input, &target)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 0.4076);

    Ok(())
}

#[test]
fn cross_entropy_rejects_invalid_targets() -> Result<()> {
    let cpu = Device::Cpu;
    let input = Tensor::new(&[[1f32, 2., 0.], [0., -1., 1.]], &cpu)?;

    let target = Tensor::new(&[1f32, 2.], &cpu)?;
    assert_cross_entropy_err(&input, &target, "target dtype");

    let target = Tensor::new(&[1u32, 3], &cpu)?;
    assert_cross_entropy_err(&input, &target, "target index 3 is out of bounds");

    let target = Tensor::new(&[1i64, -1], &cpu)?;
    assert_cross_entropy_err(&input, &target, "target index -1 is negative");

    let target = Tensor::new(&[1i64, u32::MAX as i64 + 1], &cpu)?;
    assert_cross_entropy_err(&input, &target, "target index 4294967296 is out of bounds");

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn cross_entropy_metal_target_dtypes() -> Result<()> {
    let device = Device::new_metal(0)?;
    cross_entropy_device_u8_target(&device)?;
    cross_entropy_device_rejects_i64_target(&device)?;
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn cross_entropy_metal() -> Result<()> {
    let device = Device::new_metal(0)?;
    let input = Var::new(&[[1f32, 2., 0.], [0., -1., 1.]], &device)?;
    let target = Tensor::new(&[1u32, 2], &device)?;

    let loss = candle_nn::loss::cross_entropy(input.as_tensor(), &target)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 0.4076);

    let grads = loss.backward()?;
    let grad = grads.get(input.as_tensor()).unwrap();
    assert_eq!(
        to_vec2_round(grad, 4)?,
        &[[0.1224, -0.1674, 0.045], [0.1224, 0.045, -0.1674]]
    );
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn cross_entropy_metal_invalid_target_yields_nan() -> Result<()> {
    let device = Device::new_metal(0)?;
    cross_entropy_device_invalid_u32_target_yields_nan(&device)?;
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn cross_entropy_metal_matches_cpu_class_counts() -> Result<()> {
    let device = Device::new_metal(0)?;
    for n_classes in [1, 37, 1024, 1025, 2049] {
        cross_entropy_device_matches_cpu(&device, n_classes)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cross_entropy_cuda_target_dtypes() -> Result<()> {
    let device = Device::new_cuda(0)?;
    cross_entropy_device_u8_target(&device)?;
    cross_entropy_device_rejects_i64_target(&device)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cross_entropy_cuda_invalid_target_yields_nan() -> Result<()> {
    let device = Device::new_cuda(0)?;
    cross_entropy_device_invalid_u32_target_yields_nan(&device)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cross_entropy_cuda_matches_cpu_class_counts() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for n_classes in [1, 37, 1024, 1025, 2049] {
        cross_entropy_device_matches_cpu(&device, n_classes)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cross_entropy_cuda_f64() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let input = Var::new(&[[1f64, 2., 0.], [0., -1., 1.]], &device)?;
    let target = Tensor::new(&[1u32, 2], &device)?;

    let loss = candle_nn::loss::cross_entropy(input.as_tensor(), &target)?;
    assert!((loss.to_scalar::<f64>()? - 0.4076059644443803).abs() < 1e-12);

    let grads = loss.backward()?;
    let grad = grads.get(input.as_tensor()).unwrap();
    let grad = grad.to_dtype(candle::DType::F32)?;
    assert_eq!(
        to_vec2_round(&grad, 4)?,
        &[[0.1224, -0.1674, 0.045], [0.1224, 0.045, -0.1674]]
    );
    Ok(())
}

/* Equivalent python code:
import torch
import torch.nn.functional as F

inp = torch.Tensor([[ 2.3611, -0.8813, -0.5006, -0.2178],
        [ 0.0419,  0.0763, -1.0457, -1.6692],
        [-1.0494,  0.8111,  1.5723,  1.2315],
        [ 1.3081,  0.6641,  1.1802, -0.2547],
        [ 0.5292,  0.7636,  0.3692, -0.8318]])

target = torch.Tensor([[0., 1., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.],
        [0., 0., 1., 0.]])

print(F.binary_cross_entropy_with_logits(inp, target))
*/
#[test]
fn binary_cross_entropy_with_logit() -> Result<()> {
    let cpu = Device::Cpu;

    let inp = [
        [2.3611f32, -0.8813, -0.5006, -0.2178],
        [0.0419, 0.0763, -1.0457, -1.6692],
        [-1.0494, 0.8111, 1.5723, 1.2315],
        [1.3081, 0.6641, 1.1802, -0.2547],
        [0.5292, 0.7636, 0.3692, -0.8318],
    ];

    let target = [
        [0.0f32, 1., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
    ];

    let inp = Tensor::new(&inp, &cpu)?;
    let target = Tensor::new(&target, &cpu)?;

    let loss = candle_nn::loss::binary_cross_entropy_with_logit(&inp, &target)?;

    assert_eq!(to_vec0_round(&loss, 4)?, 0.8224);
    Ok(())
}

/* Equivalent python code:
import torch
import torch.nn.functional as F

inp = torch.Tensor([[ 2.3611, -0.8813, -0.5006, -0.2178],
        [ 0.0419,  0.0763, -1.0457, -1.6692],
        [-1.0494,  0.8111,  1.5723,  1.2315],
        [ 1.3081,  0.6641,  1.1802, -0.2547],
        [ 0.5292,  0.7636,  0.3692, -0.8318]])

target = torch.Tensor([[0., 1., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.],
        [0., 0., 1., 0.]])

print(F.huber_loss(inp, target))
print(F.huber_loss(inp,target,delta=0.88))
*/
#[test]
fn huber_loss() -> Result<()> {
    let cpu = Device::Cpu;
    let inp = [
        [2.3611f32, -0.8813, -0.5006, -0.2178],
        [0.0419, 0.0763, -1.0457, -1.6692],
        [-1.0494, 0.8111, 1.5723, 1.2315],
        [1.3081, 0.6641, 1.1802, -0.2547],
        [0.5292, 0.7636, 0.3692, -0.8318],
    ];

    let target = [
        [0.0f32, 1., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
    ];

    let inp = Tensor::new(&inp, &cpu)?;
    let target = Tensor::new(&target, &cpu)?;
    let loss = candle_nn::loss::huber(&inp, &target, 1.0)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 0.4734);
    let loss = candle_nn::loss::huber(&inp, &target, 0.88)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 0.4483);
    Ok(())
}
