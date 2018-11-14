To run with default parameters using example edgelist in the folder:

python metapath2vec.py -file 20170301-20170307.edgelist

==============================
All parameters:
-file           -- an edgelist input file
-walks_outfilename            --name of the generated walks file
-numwalks       --how many lines one word will start, default is 1
-walklength     --length of each line is this parameter times 5, default is 10, length of each line is 50 words 
-embedding_outfilename        --name of the generated embeddings file
-window_size                  --window size, default is 1

==============================
Input: 
an edgelist file, format example:(p_123,u_4352)  
==============================
Output:
a file containing all walks
a file containing all embeddings
a picture of loss progress     --Currently it'll add one dot each 5 steps



I haven't add other parameters to arg parser.
But you could adjust default in the model.py codes