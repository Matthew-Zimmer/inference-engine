# Inference Engine

I am building an application which can serve low and high priority embedding tasks.

## Steps

01. [x] write the zig hello world program
02. [x] write the low priority python hello program
03. [x] write the high priority python hello program
04. [x] have the main program fork and execute the low priority program
05. [x] have the main program fork and execute the high priority program
06. [x] write a shared object library for the main program and call a function from it from the main program
07. [x] have the zig create and destroy shared memory
08. [x] pass the shared fd to high priority python program
09. [x] pass the shared fd to low priority python program
10. [x] write a message to the shared memory from zig
11. [x] read the message from shared memory in the high prog
12. [x] read the message from shared memory in the low prog
13. [x] allocate the object into shared memory
14. [x] modify the runtime api to call a function on a pointer
15. [x] modify high script to show the engine value
16. [x] modify low script to show the engine value
17. [x] add common.py and sym link it to the scripts
18. [x] modify enqueue high/low request to take in engine and text
19. [x] add echo text to engine
20. [x] modify high/low scripts to use common.py 
21. [x] update common to expose new api
22. [x] update high/low to write text
23. [x] add tokenization queue to engine
24. [x] update runtime to write to tokenization queue
25. [x] add tick loop to engine
26. [x] call tick loop in main executable
27. [x] add enqueing text to engine
28. [x] have the engine write the text to the shared memory region
29. [x] add the offset to the tokenization queue
30. [x] implement the process tokenize pipeline
31. [x] clean up mess made in shared memory debugging
32. [x] add a shared thread pool to infernce engine for tokenization
33. [x] prepare the model vocab to a trie dict
34. [x] convert trie dict to flat trie 
35. [x] convert flat trie to binary file
36. [x] compile in the model vocab trie binary blob to the executable
37. [x] convert trie dict to trie_root
38. [x] implement word piece algo for the model
39. [x] print the token ids
40. [x] write tests for word piece algo
41. [ ] write the upcast u16 -> i64 cuda kernel
42. [x] wrap cuda kernel in C linkage
43. [x] wrap tensorRT functions in C linkage
44. [x] delcare cuda kernel function in zig
45. [x] delcare tensorRT functions in zig
46. [x] allocate memory on gpu and deallocate it
47. [ ] allocate the tensor pool on the gpu
48. [x] initialize the cuda streams
49. [x] initialize the execution contexts
50. [x] add a sig term + sig int handler to gracefully shutdown
51. [x] convert static library to zig module
52. [x] prepare the pytensor to onnx format
53. [x] prepare the onnx format to .engine format

