## ElementAI Design Overview
ElementAI is a Machine Learning framework written in Rust which aims to be Small and Performant while being extensible to new Backend. 
#### Key considerations for the Design
- **Familiarity:** The ElementAI API should be very easy to use and easy to use for existing PyTorch users.
- **Extensible:** It should be extensible to add new Backend without changing the Core API's.
- **Performant and Safe:** Should be performant and limited use of Unsafe Rust code. (Unsafe should be avoided at any cost and only allowed if there is no other alternate approach).
- **Small Footprint**: The Library should be readable and have small footprint for greater agility.

##### Familiarity
- The API design for Tensor, Module should be similar to PyTorch as most of the Industry is towards using PyTorch and it has a clean API Design.
- Any new improvements towards the API design which can make the usage easier is appreciated.
- Rust's Complexity should be removed from the Public facing API's.

##### Extensible
- One problem with Candle (Rust ML Framework) is that the different Backend implementations are resided in the Framework so its not extensible. i.e If any other developer wants to add a new Backend it should be easy to do so.
- Backend functionalities should be exposed as Rust Traits so that new Backend developers can readily use them.
- The reason for being extensible and small is to have rapid research towards new AI Accelerated hardware like NPU (Neural Processing Unit)


### API Design
- Tensor

```rust
use element_core::{Tensor, Backend, CpuBackend, Device};

fn main(){
	type B = CpuBackend;
	let device = Default::default(); //CpuDevice
	let tensor1 = Tensor<B>::zeros((1, 2), DType::F32, &device);
	let tensor2 = Tensor<B>::ones((1, 2), DType::F64, &device);
	let tensor3 = tensor1 + tensor2;
	println!("{}", tensor3);	
	let tensor4 = Tensor::<B>::arange(0..5, &device);
	let tensor5 = Tensor::<B>::new(vec![4.0, 5.0], &device);
	let tensor6 = Tensor::<B>::new(&[[1.0, 2.0]], &deivce);

	//This is not needed - as writing None is of no use, if we had
	// Defaulting the params and I didn't need to write None when calling the
	// Function it would have be great to deafult
	let tensor6 = Tensor::<B>::new(&[[1.0, 2.0]], None); //Default device taken

	//Here the data is f64 and when passed in DType we want to convert
	// The data type to provided one as it will be helpful at times
	let tensor7 = Tensor::<B>::new(&[1.0, 2.0], DType::F32, &device); //DType param is Option<DType>
	
}
```

##### Open Questions
- Should we add a layer of StridedIndex or Indexing functionality in the Tensor itself or leave it to the Backend? Or Should we create another Crate for extending the Indexing which Backend can utilise?
- How do we go with Compiling or Dynamic Graph Execution?