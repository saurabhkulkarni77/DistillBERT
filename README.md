Distilling BERT is a 5 stage process:
1. Download Pretrained Model

      In order to distill knowledge from a large pretrained transformer model, need to first download that model.
      I will assume you have downloaded BERT-Base Uncased (12-layer, 768-hidden, 12-heads, 110M parameters )within this repository.

2. Extract Wikipedia
      
      Go to https://dumps.wikimedia.org/ for a wide variety of dumps of Wikipedia's website. I prefer https://dumps.wikimedia.org/enwiki/latest/ for the latest dump.

      When working on my machine I prefer a smaller shard: enwiki-latest-pages-articles1.xml-p10p30302.bz2
      
      Run python WikiExtractor.py /dumppath/your_chosen_dump.bz2 --output /outputdir/ --json 
      
      Run extract_jsons.py 
      
      python extract_jsons.py --folder /folderof/foldersof/jsons/ --write_file /filepath/write_file.txt 
      
      And with that you have the data txt file you need for training!

3. Prepare Text For TensorFlow
      we must first slit this file into smaller ones in order not to run into RAM or disk space problems later down the line. 
      
      Thus we must run split_text.py
      
      python split_text.py --read_file wikipedia.txt --split_number 20 --folder data/split_dir --name_base wiki_split
      
      After splitting Wikipedia into smaller txt files, we can turn all of them into tfrecord files by running multifile_create_pretraining_data.py
      
      python multifile_create_pretraining_data.py \
    --input_dir data/split_dir/ \
    --output_dir data/record_intermed \
    --output_base_name wiki_intermed \
    --vocab_file uncased_L-12_H-768_A-12/vocab.txt
    
4. Extract Teacher Neural Network Outputs
      
      One possibility for performing knowledge distillation is to pass an input to the student and teacher networks at the same time and using the outputs of the teacher for the student to learn from. However, considering that this will put a strain on our RAM and that we will be making multiple runs through each of over our data, it is more resource efficient to run through all of our data once and save the output of our teacher network with the inputs that were fed to it. This is accomplished by running extract_teacher_labels_truncated.py
      
      python extract_teacher_labels_truncated.py \
    --bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
    --data/record_intermed/wiki_intermed_0.tfrecord \
    --output_file data/record_distill/wiki_distill_0.tfrecord \
    --truncation_factor 10 \
    --init_checkpoint uncased_L-12_H-768_A-12/bert_model.ckpt 
5. Distill Knowledge 
    
    Now that we have our teacher outputs we can start training a student network! To run on a single machine run network_distillation_single_machine_truncated.py
    
    python network_distillation_single_machine_truncated.py \
    --bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
    --input_file data/record_distill/wiki_distill_0.tfrecord \
    --output_dir output_dir \
    --truncation_factor 10 \
    --do_train True \
    --do_eval true  
6. Single-Node Distributed Distillation
    
    
    Now suppose you have a lil cluster of 8 GPU's! If you have Horovod installed, you can perform some distributed training!!! (If you don't have horovod installed you can install it here). We shall run network_distillation_distributed_truncated.py to perform distributed training as such:
    
    

    mpirun -np 8 \
    -H localhost:8 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python network_distillation_distributed_truncated.py \
    --bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
    --input_file data/record_distill/wiki_distill_0.tfrecord \
    --output_dir output_dir \
    --truncation_factor 10 \
    --do_train True \
    --do_eval true
