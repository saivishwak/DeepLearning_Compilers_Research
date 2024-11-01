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

Here the primitive is the `TensorKind`, So if we initialize a Float Tensor then the primitive here is `FloatTensorPrimitive`, if we initialise IntTensor then its `IntTensorPrimitive`.

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
Tensor is parameterized with `TensorKind` Parameter, and each `TensorKind` parameter (Float, Int, Bool) has their Ops implementation defined with Traits like, `BasicOps`, `Numeric` etc.

One awesome point to not is, Tensor/base.rs file does not have the ones and zeros implementations, These are coming from the Numeric implementations and the base.rs has only `BasicOps` defined.

### Things I believe can be better
 I am right now against of using Three different kinds of Tensors, Rather prefer to stick to PyTorch implementation of passing `DType` explicitly

```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)```


```rust
use burn::tensor::Tensor;

fn main() {
    type B = burn::backend::NdArray;
    let device = Default::default();
    let tensor1 = Tensor::<B, 1>::from_floats([1], &device);
    // If you see here I am giving 1 as the value which is a i64 in rust but because I am calling from_floats the data is converted, This ambiguity is bad. The tensor should have clearly defined data type like PyTorch.
    // Also do we really need to Dimension to be given as a type parameter or can we infer it directly from the data and shape given?
    println!("{}", tensor1);
}
```

```rust
use burn::tensor::Tensor;

fn main() {
    type B<Elem> = burn::backend::NdArray<Elem>;
    let device = Default::default();
    let tensor1 = Tensor::<B<f32>, 1>::from_data(vec![1.0], &device);
    //here vec is not supported, I feel for ease of reserach giving support for nested vec is good
    println!("{}", tensor1);
}
```

```rust
use burn::tensor::Tensor;

fn main() {
    type B = burn::backend::NdArray;
    let device = Default::default();
    let tensor1 = Tensor::<B, 1>::from_data([1i32], &device);
    println!("{}", tensor1);
}
```

Output:
```shell
Tensor {
  data:
[1.0],
  shape:  [1],
  device:  Cpu,
  backend:  "ndarray",
  kind:  "Float",
  dtype:  "f32",
}
```

I have given i32 as the data but its internally converted back to f32 as it is the default Element type for Backends. It is better to avoid these ambiguities so that mistakes are avoided and we utilized the power of Rust Type Safety in a better way.

```rust
use burn::tensor::{Tensor, TensorData};

fn main() {
    type B = burn::backend::NdArray;
    let device = Default::default();
    let tensor1 = Tensor::<B, 1>::from_data(TensorData::from([1i32, 2, 3]), &device);
    println!("{}", tensor1);
}
```

Output:
```shell
Tensor {
  data:
[1.0, 2.0, 3.0],
  shape:  [3],
  device:  Cpu,
  backend:  "ndarray",
  kind:  "Float",
  dtype:  "f32",
}
```

Here also we have same auto type conversion ambiguity.

```rust
use burn::tensor::{Int, Tensor};

fn main() {
    type B = burn::backend::NdArray;
    let device = Default::default();
    let tensor1 = Tensor::<B, 1, Int>::from_data([1.0], &device);
    println!("{}", tensor1);
}
```

Output:
```shell
Tensor {
  data:
[1],
  shape:  [1],
  device:  Cpu,
  backend:  "ndarray",
  kind:  "Int",
  dtype:  "i64",
}
```

Similar ambiguity for IntTensor given a float element.


##### Type Conversions
```rust
use burn::tensor::{Int, Tensor};

fn main() {
    type B = burn::backend::NdArray<f64>; //If we dont give f64, default is f32
    let device = Default::default();
    let tensor1 = Tensor::<B, 1, Int>::from_data([1], &device).float();
    println!("{}", tensor1);
}
```

```rust
use burn::tensor::{Int, Tensor};

fn main() {
    type B = burn::backend::NdArray<f64>;
    let device = Default::default();
    let tensor1 = Tensor::<B, 1>::from_data([1], &device).int();
    println!("{}", tensor1);
}
```

Here as well the `int()` converts the tensor to `i64` datatype, there is no control to convert to other data types as far as I have used it.

I believe like PyTorch giving ability to cast to different types with `to_dtype()` or `to()` methods which gives much flexibility and also provide default `.float()`, `.int()`, `.double()` to convert to basic types.
#### Burn (Initial Analysis)
- https://github.com/tracel-ai/burn/tree/7389ef20b009410464065841ff3b4821a7350a2c

Main concepts
- Backend
- Tensor
- Data
- Ops
- Graph

##### Tensor
```rust
/// A tensor or a *n-dimensional* array.
#[derive(Debug, Clone)]
pub struct Tensor<B: Backend, const D: usize> {
    pub(crate) value: B::TensorPrimitive<D>,
}

impl<const D: usize, B> Tensor<B, D>
where
    B: Backend,
{
    pub(crate) fn new(tensor: B::TensorPrimitive<D>) -> Self {
        Self { value: tensor }
    }

    pub fn reshape<const D2: usize>(&self, shape: Shape<D2>) -> Tensor<B, D2> {
        Tensor::new(self.value.reshape(shape))
    }
	
	/// Returns the data of the current tensor without taking ownership.
    pub fn to_data(&self) -> Data<B::Elem, D> {
        self.value.to_data()
    }

    /// Create a tensor from the given data.
    pub fn from_data(data: Data<B::Elem, D>) -> Self {
        let tensor = B::from_data(data, B::Device::default());
        Tensor::new(tensor)
    }
	
	pub fn add(&self, other: &Self) -> Self {
        Self::new(self.value.add(&other.value))
    }
	
	pub fn zeros(shape: Shape<D>) -> Self {
        let tensor = B::zeros(shape, B::Device::default());
        Self::new(tensor)
    }

    /// Create a tensor of the given shape where each element is one.
    pub fn ones(shape: Shape<D>) -> Self {
        let tensor = B::ones(shape, B::Device::default());
        Self::new(tensor)
    }
}
```


###### Backend
```rust
pub trait Backend: Clone + Sized + Default + Send + Sync + std::fmt::Debug + 'static {
    type Device: Copy + Clone + Default + std::fmt::Debug + Send + Sync;
    type Elem: Element;
    type FullPrecisionElem: Element;
    type FullPrecisionBackend: Backend<Elem = Self::FullPrecisionElem, Device = Self::Device>;
    type IntegerBackend: Backend<Elem = i64, Device = Self::Device>;
    type TensorPrimitive<const D: usize>: TensorOpsUtilities<Self::Elem, D>
        + TensorOpsMatmul<Self::Elem, D>
        + TensorOpsTranspose<Self::Elem, D>
        + TensorOpsMul<Self::Elem, D>
        + TensorOpsDiv<Self::Elem, D>
        + TensorOpsNeg<Self::Elem, D>
        + TensorOpsAdd<Self::Elem, D>
        + TensorOpsSub<Self::Elem, D>
        + TensorOpsDetach<Self::Elem, D>
        + Zeros<Self::TensorPrimitive<D>>
        + Ones<Self::TensorPrimitive<D>>
        + TensorOpsReshape<Self, D>
        + TensorOpsPrecision<Self, D>
        + TensorOpsDevice<Self, D>
        + TensorOpsIndex<Self::Elem, D>
        + TensorOpsAggregation<Self, D>
        + TensorOpsExp<Self::Elem, D>
        + TensorOpsArg<Self, D>
        + TensorOpsCat<Self::Elem, D>
        + TensorOpsLog<Self::Elem, D>
        + TensorOpsErf<Self::Elem, D>
        + TensorOpsPow<Self::Elem, D>
        + TensorOpsMask<Self, D>
        + TensorOpsMapComparison<Self, D>
        + ReLU<Self::Elem, D>
        + Clone
        + Send
        + Sync
        + Send
        + Sync
        + 'static
        + std::fmt::Debug;

    type BoolTensorPrimitive<const D: usize>: TensorOpsUtilities<bool, D>
        + Clone
        + Send
        + Sync
        + 'static
        + std::fmt::Debug;

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: Self::Device,
    ) -> Self::TensorPrimitive<D>;

    fn from_data_bool<const D: usize>(
        data: Data<bool, D>,
        device: Self::Device,
    ) -> Self::BoolTensorPrimitive<D>;

    fn ad_enabled() -> bool;
    fn name() -> String;
    fn seed(seed: u64);

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<Self::Elem>,
        device: Self::Device,
    ) -> Self::TensorPrimitive<D>;

    fn zeros<const D: usize>(shape: Shape<D>, device: Self::Device) -> Self::TensorPrimitive<D> {
        Self::from_data(Data::zeros(shape), device)
    }

    fn ones<const D: usize>(shape: Shape<D>, device: Self::Device) -> Self::TensorPrimitive<D> {
        Self::from_data(Data::ones(shape), device)
    }
}

```
### GIT
- https://github.com/tracel-ai/burn/blob/3ba6c698755984d24d43ded235f6aab693908f4c/burn-tensor/src/tensor/data.rs
- https://github.com/tracel-ai/burn/tree/7389ef20b009410464065841ff3b4821a7350a2c