### Architecture Overview

Burn aims to be very flexible so new Backend's can be easily configured and also being performant as the same time.

To get the flexibility Burn extensively uses the Rusts powerful Trait feature to allow what operations that the Backend should implement.

Example Usage:
```rust
// Initialization from a given Backend (Wgpu)
let tensor_1 = Tensor::<Wgpu, 1>::from_data([1.0, 2.0, 3.0], &device);

// Initialization from a generic Backend
let tensor_2 = Tensor::<Backend, 1>::from_data(TensorData::from([1.0, 2.0, 3.0]), &device);

// Initialization using from_floats (Recommended for f32 ElementType)
// Will be converted to TensorData internally.
let tensor_3 = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);
```

Burn is using `TensorData` struct to convert data types, but it internally uses bytemuck which has usafe rust code. (We need to avoid that!)
https://github.com/tracel-ai/burn/blob/main/crates/burn-tensor/src/tensor/data.rs#L94C9-L94C17


#### How Tensor ops like zeros, ones, reshape and Basic ops work in Burn?

```rust
#[derive(new, Clone, Debug)]
pub struct Tensor<B, const D: usize, K = Float>
where
    B: Backend,
    K: TensorKind<B>,
{
    pub(crate) primitive: K::Primitive<D>,
}
```

The Tensor Struct has Type Bounds with Backend, TensorKind and Dimension. 

There are basically Three Types of Tensors that can be created in Burn, Float, Bool and Int.

```rust
use crate::backend::Backend;

/// A type-level representation of the kind of a float tensor
#[derive(Clone, Debug)]
pub struct Float;

/// A type-level representation of the kind of a int tensor.
#[derive(Clone, Debug)]
pub struct Int;

/// A type-level representation of the kind of a bool tensor.
#[derive(Clone, Debug)]
pub struct Bool;

#[derive(Debug, Clone)]
/// A primitive tensor representation.
pub enum TensorPrimitive<B: Backend, const D: usize> {
    /// Float tensor primitive.
    Float(B::FloatTensorPrimitive<D>),
    /// Quantized float tensor primitive.
    QFloat(B::QuantizedTensorPrimitive<D>),
}

impl<B: Backend, const D: usize> TensorPrimitive<B, D> {
    /// Returns the full tensor representation.
    pub fn tensor(self) -> B::FloatTensorPrimitive<D> {
        match self {
            Self::QFloat(tensor) => B::dequantize(tensor),
            Self::Float(tensor) => tensor,
        }
    }
}

/// A type-level representation of the kind of a tensor.
pub trait TensorKind<B: Backend>: Clone + core::fmt::Debug {
    /// The primitive type of the tensor.
    type Primitive<const D: usize>: Clone + core::fmt::Debug + Send;

    /// The name of the tensor kind.
    fn name() -> &'static str;
}

impl<B: Backend> TensorKind<B> for Float {
    type Primitive<const D: usize> = TensorPrimitive<B, D>;
    fn name() -> &'static str {
        "Float"
    }
}
```


We have implementation of Tensor with BasicOps as K

```rust
impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
{
}
```

This implementation has methods like shape, empty reshape, transpose etc

```rust
/// Trait that list all operations that can be applied on all tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait BasicOps<B: Backend>: TensorKind<B> {
	type Elem = Element;
}
```

```rust
/// Element trait for tensor.
pub trait Element:
    ToElement
    + ElementRandom
    + ElementConversion
    + ElementPrecision
    + ElementComparison
    + bytemuck::CheckedBitPattern
    + bytemuck::NoUninit
    + core::fmt::Debug
    + core::fmt::Display
    + Default
    + Send
    + Sync
    + Copy
    + 'static
{
    /// The dtype of the element.
    fn dtype() -> DType;
}
```

Implementation of BasicOps for Float TensorKind

```rust
impl<B: Backend> BasicOps<B> for Float {
    type Elem = B::FloatElem;
}
```

The methods in here all mostly call the Backend functions using `B::float_shape()` the reason for **BasicOps** as a wrapper is to use for **float**, **int** and **bool** conversions of data types.

So, shape in Int Basic ops calls `B::int_shape()` instead of the float method

```rust
pub trait Backend:
    FloatTensorOps<Self>
    + BoolTensorOps<Self>
    + IntTensorOps<Self>
    + ModuleOps<Self>
    + ActivationOps<Self>
    + QTensorOps<Self>
    + Clone
    + Sized
    + Default
    + Send
    + Sync
    + core::fmt::Debug
    + 'static
{
    /// Device type.
    type Device: DeviceOps;

    /// A bridge that can cast tensors to full precision.
    type FullPrecisionBridge: BackendBridge<Self> + 'static;

    /// Tensor primitive to be used for all float operations.
    type FloatTensorPrimitive<const D: usize>: Clone + Send + 'static + core::fmt::Debug;
    /// Float element type.
    type FloatElem: Element;

    /// Tensor primitive to be used for all int operations.
    type IntTensorPrimitive<const D: usize>: Clone + Send + 'static + core::fmt::Debug;
    /// Int element type.
    type IntElem: Element;

    /// Tensor primitive to be used for all bool operations.
    type BoolTensorPrimitive<const D: usize>: Clone + Send + 'static + core::fmt::Debug;

    /// Tensor primitive to be used for all quantized operations.
    type QuantizedTensorPrimitive<const D: usize>: QTensorPrimitive
        + Clone
        + Send
        + 'static
        + core::fmt::Debug;

    /// If autodiff is enabled.
    fn ad_enabled() -> bool {
        false
    }

    /// Name of the backend.
    fn name() -> String;

    /// Seed the backend.
    fn seed(seed: u64);

    /// Sync the backend, ensure that all computation are finished.
    fn sync(_device: &Self::Device, _sync_type: SyncType) {}
}
```


###### How is Device Ordinal Used?
```rust
/// The device id.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, new)]
pub struct DeviceId {
    /// The type id identifies the type of the device.
    pub type_id: u16,
    /// The index id identifies the device number.
    pub index_id: u32,
}

/// The handle device trait allows to get an id for a backend device.
pub trait DeviceOps: Clone + Default + PartialEq + Send + Sync + core::fmt::Debug {
    /// Return the [device id](DeviceId).
    fn id(&self) -> DeviceId;
}
```

The `DeviceOps` needs to be implemented for each Backend, where the `type_id` is used for the different Devices a Backend can have like in Candle we have CPU, CUDA and Metal devices supported. So each Backend have multiple devices.

Burn uses Default for Device for Backend

```rust

impl Default for CandleDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

fn example(){
	let device = Default::default();
    let tensor = Tensor::<B, 3>::ones(Shape::new([2, 1, 4]), &device);
}
```

###### How is Indexing implemented?
Indexing is not directly implemented as the Backend will have Tensor Ops already defined and we directly call them. It will be a main Job is custom Backend is implemented.

###### How is Storage used?

```rust
#[derive(new, Clone, Debug)]
pub struct Tensor<B, const D: usize, K = Float>
where
    B: Backend,
    K: TensorKind<B>,
{
    pub(crate) primitive: K::Primitive<D>,
}

```

Here the primitive is the TensorKind, So if we initialise a Float Tensor then the primitive here is FloatTensorPrimitive, if we initialise IntTensor then its IntTensorPrimitive.

Sneak peek into Backend trait (Below is only a portion of the code)
 ```rust
pub trait Backend:
    FloatTensorOps<Self>
    + BoolTensorOps<Self>
    + IntTensorOps<Self>
    + ModuleOps<Self>
    + ActivationOps<Self>
    + QTensorOps<Self>
    + Clone
    + Sized
    + Default
    + Send
    + Sync
    + core::fmt::Debug
    + 'static
{
    /// Device type.
    type Device: DeviceOps;

    /// Tensor primitive to be used for all float operations.
    type FloatTensorPrimitive<const D: usize>: Clone + Send + 'static + core::fmt::Debug;

    /// Tensor primitive to be used for all int operations.
    type IntTensorPrimitive<const D: usize>: Clone + Send + 'static + core::fmt::Debug;
}
```


###### Ops Implementation
Tensor is Parameterised with TensorKind Parameter, and each TensorKind parameter (Float, Int, Bool) has their Ops implementation defined with Traits like, BasicOps, Numeric etc.

One awesome point to not is, Tensor/base.rs file does not have the ones and zeros implementations, These are coming from the Numeric implementations and the base.rs has only BasicOps defined.

#### Things that can be better
- I am right now against of using Three different kinds of Tensors, Rather prefer to stick to PyTorch implementation of passing DType explicitly

```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
```


###### GIT
- https://github.com/tracel-ai/burn/blob/3ba6c698755984d24d43ded235f6aab693908f4c/burn-tensor/src/tensor/data.rs