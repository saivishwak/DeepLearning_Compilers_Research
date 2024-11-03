[ArXiV](https://arxiv.org/pdf/2002.03794)

#### COMMON DESIGN ARCHITECTURE OF DL COMPILERS
The common design architecture of a DL compiler primarily contains two parts: the compiler frontend and the compiler backend, as shown in Figure 2. The intermediate representation (IR) is spread across both the frontend and the backend. Generally, IR is an abstraction of the program and is used for program optimizations. Specifically, the DL models are translated into multi-level IRs in DL compilers, where the high-level IR resides in the frontend, and the low-level IR resides in the backend. Based on the high level IR, the compiler frontend is responsible for hardware-independent transformations and optimizations. Based on the low-level IR, the compiler backend is responsible for hardware-specific optimizations, code generation, and compilation.

The high-level IR, also known as graph IR, represents the computation and the control flow and is hardware-independent. **The design challenge of high-level IR is the ability of abstraction of the computation and the control flow, which can capture and express diverse DL models.** The goal of the high-level IR is to establish the control flow and the dependency between the operators and the data, as well as provide an interface for graph-level optimizations.

The low-level IR is designed for hardware-specific optimization and code generation on diverse hardware targets. Thus, the low-level IR should be fine-grained enough to reflect the hardware characteristics and represent the hardware-specific optimizations. It should also allow the use of mature third-party tool-chains in compiler backends such as Halide [ 77], polyhedral model [31], and LLVM [51]. The detailed discussion of low-level IR is presented in Section 4.2.

The frontend takes a DL model from existing DL frameworks as input, and then transforms the model into the computation graph representation (e.g., graph IR). The frontend needs to implement various format transformations To support the diverse formats in different frameworks. The computation graph optimizations incorporate the optimization techniques from both general-purpose compilers and the DL specific optimizations, which reduce the redundancy and improve the efficiency upon the graph IR. Such optimizations can be classified into node-level (e.g., nop elimination and zero-dim-tensor elimination), block-level (e.g., algebraic simplification, operator fusion, and operator sinking) and dataflow-level (e.g., CSE, DCE, static memory planning, and layout transformation). After the frontend, the optimized computation graph is generated and passed to the backend.

The backend transforms the high-level IR into low-level IR and performs hardware-specific optimizations. On the one hand, it can directly transform the high-level IR to third-party tool-chains such as LLVM IR to utilize the existing infrastructures for general-purpose optimizations and code generation. On the other hand, it can take advantage of the prior knowledge of both DL models and hardware characteristics for more efficient code generation, with customized compilation passes. The commonly applied hardware-specific optimizations include hardware intrinsic mapping, memory allocation and fetching, memory latency hiding, parallelization as well as loop oriented optimizations. To determine the optimal parameter setting in the large optimization
space, two approaches are widely adopted in existing DL compilers such as auto-scheduling (e.g., polyhedral model) and auto-tuning (e.g., AutoTVM). The optimized low-level IR is compiled using JIT or AOT to generate codes for different hardware targets.

#### KEY COMPONENTS OF DL COMPILERS
##### High-level IR

**DAG-based IR** - DAG-based IR is one of the most traditional ways for the compilers to build a computation graph, with nodes and edges organized as a directed acyclic graph (DAG). In DL compilers [ 17, 21, 53 , 79, 91 ], the nodes of a DAG represent the atomic DL operators (convolution, pooling, etc.), and the edges represent the tensors. And the graph is acyclic without loops, which differs from the data dependence graphs [ 50] (DDG) of generic compilers [51 , 52]. And with the help of the DAG computation graph, DL compilers can analyze the relationship and dependencies between various operators and use them to guide the optimizations.

**Let-binding-based IR** - Let-binding is one method to solve the semantic ambiguity by offering let expression to certain functions with restricted scope used by many high-level programming languages such as Javascript [30], F# [ 76], and Scheme [ 3 ]. When using the let keyword to define an expression, a let node is generated, and then it points to the operator and variable in the expression instead of just building computational relation between variables as a DAG. TVM adopts both DAG-based IR and let-binding-based IR to obtain the benefits of both.

**Representing Tensor Computation** - 
**Function-based** - XLA, Glow, nGraph
**Lambda expression** - TVM
**Einstein notation**

###### Implementation of Graph IR
**Data representation** - The data in DL compilers (e.g., inputs, weights, and intermediate data) are usually organized in the form of tensors, which are also known as multi-dimensional arrays. The DL compilers can represent tensor data directly by memory pointers, or in a more flexible way by placeholders.

**Placeholder** - Placeholder is widely used in symbolic programming (e.g., Lisp [65], Tensor- flow [1]). A placeholder is simply a variable with explicit shape information (e.g., size in each dimension), and it will be populated with values at the later stage of the computation. It allows the programmers to describe the operations and build the computation graph without concerning the exact data elements, which helps separate the computation definition from the exact execution in DL compilers.

**Unknown (Dynamic) shape representation** - The unknown dimension size is usually supported when declaring the placeholders. For instance, TVM uses Any to represent an unknown dimension (e.g.,Tensor ⟨(Any, 3), f p32⟩); XLA uses None to achieve the same purpose (e.g., t f .placeholder (“float”, [None, 3])). nGraph uses its PartialShape class. The unknown shape representation is necessary to support the dynamic model. However, to fully support dynamic model, the bound inference and dimension checking should be relaxed.

**Data layout** - The data layout describes how a tensor is organized in memory, and it is usually a mapping from logical indices to memory indices. The data layout usually includes the sequence of dimensions (e.g., NCHW and NHWC), tiling, padding, striding, etc. **TVM and Glow represent data layout as operator parameters and require such information for computation and optimization. However, combining data layout information with operators rather than tensors enables intuitive implementation for certain operators and reduces the compilation overhead. XLA represents data layout as constraints related to its backend hardware. Relay and MLIR are going to add data layout information into their type systems for tensors.**

**Control flow** - Control flow is needed when representing complex and flexible models. Models such as RNN and Reinforcement learning (RL) depend on recurrent relations and data-dependent conditional execution [103], which requires control flow. Without supporting control flow in graph IR of DL compilers, these models must rely on the control flow support of the host languages (e.g., if and while in Python) or static unrolling, which deteriorates the computation efficiency.

**Derivative** - The derivative operator of an operator Op takes the output gradients and the input data of Op as its inputs, and then calculates the gradient of Op.

**Customized operators** - It allows programmers to define their operators for a particular purpose. Providing support for customized operators improves the extensibility of DL compilers. For example, when defining new operators in Glow, the programmers need to realize the logic and node encapsulation. In addition, extra efforts are needed, such as the lowering step, operation IR generation, and instruction generation, if necessary. Whereas, TVM and TC require less programming efforts except describing the computation implementation. Specifically, the users of TVM only need to describe the computation and the schedule and declare the shape of input/output tensors.

Nearly all DL compilers have their unique high-level IRs. However, they share
similar design philosophies, such as using DAG and let-binding to build the computation graph. In addition, they usually provide convenient ways for programmers to represent tensor computation. The data and operators designed in high-level IRs are flexible and extensible enough to support diverse DL models. More importantly, the high-level IRs are hardware-independent and thus can be applied with different hardware backend.

#### Low-level IR

Low-level IR describes the computation of a DL model in a
more fine-grained representation than that in high-level IR, which enables the target-dependent optimizations by providing interfaces to tune the computation and memory access.

**Halide-based IR**

Halide is firstly proposed to parallelize image processing, and it is proven to be extensible and efficient in DL compilers (e.g., TVM). The fundamental philosophy of Halide is the separation of computation and schedule. Rather than giving a specific scheme directly, the compilers adopting Halide try various possible schedule and choose the best one.

TVM has improved Halide IR into an independent symbolic IR by following efforts. It removes the dependency on LLVM and refactors the structure of both the project module and the IR design of Halide, pursuing better organization as well as accessibility for graph IR and frontend language such as Python. The re-usability is also improved, with a runtime dispatching mechanism implemented to add customized operators conveniently. TVM simplifies the variable definition from string matching to pointer matching, guaranteeing that each variable has a single define location (static single-assignment, SSA) [22]).

**Polyhedral-based IR**

The polyhedral model is an important technique adopted in DL compilers.
It uses linear programming, affine transformations, and other mathematical methods to optimize loop-based codes with static control flow of bounds and branches. In contrast to Halide, the boundaries of memory reference and loop nests can be polyhedrons with any shapes in the polyhedral model. Such flexibility makes polyhedral models widely used in generic compilers.

**Code Generation based on Low-Level IR**

The low-level IR adopted by most DL compilers can be eventually lowered to LLVM IR, and benefits from LLVM’s mature optimizer and code generator.
Furthermore, LLVM can explicitly design custom instruction sets for specialized accelerators from scratch. However, traditional compilers may generate poor code when passed directly to LLVM IR. In order to avoid this situation, two approaches are applied by DL compilers to achieve hardware dependent optimization: 1) perform target-specific loop transformation in the upper IR of LLVM (e.g., Halide-based IR and polyhedral-based IR), and 2) provide additional information about the hardware target for the optimization passes. Most DL compilers apply both approaches, but the emphasis is different.

##### Conclusion

**Dynamic shape and pre/post processing**
Dynamic model becomes more and more popular in the field of DL, whose input shape or even model itself may change during execution. Particularly, in the area of NLP, models may accept inputs of various shapes, which is challenging for DL compilers since the shape of data is unknown until runtime. Existing DL compilers require more research efforts to support dynamic shape efficiently for emerging dynamic models. In addition, as future DL models become more complex, their entire control flow may inevitably include complicated pre/post-processing procedures. Currently, most DL compilers use Python as their programming language, the pre/post-processing could become a performance bottleneck when it is executed by the Python interpreter. Such potential performance bottleneck has not yet been considered by existing DL compilers. Supporting the entire control flow in DL compiler enables express and optimize the pre/post-processing along with DL models, which opens up new opportunities for performance acceleration in model deployment.

**Polyhedral model**
It is a promising research direction to combine polyhedral model and auto-tuning techniques in the design of DL compilers for efficiency. On one hand, the auto-tuning can be applied to minimize the overhead of polyhedral JIT compilation by reusing the previous configurations. On the other hand, the polyhedral model can be used to perform auto-scheduling, which can reduce the search space of auto-tuning.

**Subgraph partitioning**
DL compilers supporting subgraph partitioning can divide the computation graph into several subgraphs, and the subgraphs can be processed in different manners. The subgraph partitioning presents more research opportunities for DL compilers.

**Differentiable programming**
Differentiable programming is a programming paradigm, where the programs are differentiable thoroughly. Algorithms written in differentiable programming paradigm can be automatically differentiated, which is attractive for DL community. Many compiler projects have adopted differentiable programming, such as Myia [ 89], Flux [40] and Julia [ 13]. Unfortunately, there is little support for differential programming in existing DL compilers.

**Training support** 
In general, the model training is far less supported in current DL compilers. nGraph only supports training on the Intel NNP-T accelerator, TC only supports the auto differentiation of a single kernel, Glow has experimental training support for limited models, the training support of TVM is under development, while XLA relies on the training support of TensorFlow. In sum, current DL compilers mainly focus on bridging the gap of deploying DL models onto diverse hardware efficiently, and thus they choose inference as their primary optimization targets. However, expanding the capability of DL compilers to support model training would open up a large body of research opportunities such as optimization of gradient operators and high-order auto differentiation.
### Key Points
- The design challenge of high-level IR is the ability of abstraction of the computation and the control flow, which can capture and express diverse DL models.
- The frontend takes a DL model from existing DL frameworks as input, and then transforms the model into the computation graph representation (e.g., graph IR). The frontend needs to implement various format transformations To support the diverse formats in different frameworks.
- High Level IR Needs to be able to capture All Tensor Related Operations and Datatypes
- TVM Extended Halide IR to be more generic and better suited for languages like Python
- Difference between Rectangular and Non-Rectangular data. (Rectangular are in table form and Non Rectangular are like Lists, Graphs etc)
- Nearly all DL compilers have their unique high-level IRs. However, they share similar design philosophies, such as using DAG and let-binding to build the computation graph. In addition, they usually provide convenient ways for programmers to represent tensor computation. The data and operators designed in high-level IRs are flexible and extensible enough to support diverse DL models. More importantly, the high-level IRs are hardware-independent and thus can be applied with different hardware backend.
- The unknown dimension size is usually supported when declaring the placeholders. For instance, TVM uses Any to represent an unknown dimension (e.g.,Tensor ⟨(Any, 3), f p32⟩); XLA uses None to achieve the same purpose (e.g., t f .placeholder (“float”, [None, 3])).
- In General, on CPU, TVM and nGraph achieve better performance across all models than other DL compilers.
- nGaph achieves a better performance of the convolution layers on CPU.
- TVM performs better on GPU across these compilers.
- Dynamic Shapes are important for newer models architectures.
- Subgraph partitioning might help in performance gains with different subgraph optimizers.
- Training support is needed.