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
41. [x] write the upcast u16 -> i64 cuda kernel
42. [x] wrap cuda kernel in C linkage
43. [x] wrap tensorRT functions in C linkage
44. [x] delcare cuda kernel function in zig
45. [x] delcare tensorRT functions in zig
46. [x] allocate memory on gpu and deallocate it
47. [x] allocate the tensor pool on the gpu
48. [x] initialize the cuda streams
49. [x] initialize the execution contexts
50. [x] add a sig term + sig int handler to gracefully shutdown
51. [x] convert static library to zig module
52. [x] prepare the pytensor to onnx format
53. [x] prepare the onnx format to .engine format
54. [x] enqueue the embeding portion to the pipeline
55. [x] implement new encoder struct
56. [x] implement large chunking strategy
57. [x] implement small chunking strategy
58. [x] implement page chunking strategy
59. [ ] need to bookkeep chunk to page mapping
60. [ ] implement non chunked word peice encoder
61. [ ] need to be able to request low priority non chunked encodings
62. [ ] need to be able to request high priority non chunked encodings
63. [x] need to have an event file descriptor allocated to a request
64. [x] need to pool of event file descriptors
65. [x] after request is done need to deinit the memory
66. [ ] implement the high priority http web server
67. [ ] benchmark the inference engine
68. [x] need an averager cuda kernel on the gpu side to average all embeddings
69. [x] allocate space on gpu for u16 and u64 for tokens
70. [x] invoke the u16 -> u64 upcast kernel before sending to model
71. [x] invoke the averaging kernel after the model is ran
72. [x] change tokening cpu side to use u16s instead of u64s
73. [x] write a public zig function to inspect a chunked request event fds
74. [x] write a public zig function to inspect all event fds
75. [x] write a public zig function to inspect the chunked request result

