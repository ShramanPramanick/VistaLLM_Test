This codebase is based on the Shikra repository. https://github.com/shikras/shikra
We thank the authors for open-sourcing their work. 
We will publicly release a cleaned version of our codebase and collected datasets upon acceptance of our manuscript. 


Training Command: 
torchrun --nproc_per_node 2 --nnodes 1 mllm/pipeline/finetune.py config/shikra_pretrain_concat8_stage1.py --cfg-options model_args.model_name_or_path=<Vicuna_Path> 
--output_dir <Output_Dir>


REC Evaluation:


accelerate launch --num_processes 4 mllm/pipeline/finetune.py config/shikra_eval_multi_rec.py --cfg-options model_args.model_name_or_path=<Trained_Model_Path>
--output_dir <Output_Dir>


RES Evaluation:


accelerate launch --num_processes 4 mllm/pipeline/finetune.py config/shikra_eval_multi_res.py --cfg-options model_args.model_name_or_path=<Trained_Model_Path>
--output_dir <Output_Dir>




GREC Evaluation:


accelerate launch --num_processes 4 mllm/pipeline/finetune.py config/shikra_eval_multi_grec.py --cfg-options model_args.model_name_or_path=<Trained_Model_Path>
--output_dir <Output_Dir>




GRES Evaluation:


accelerate launch --num_processes 4 mllm/pipeline/finetune.py config/shikra_eval_multi_gres.py --cfg-options model_args.model_name_or_path=<Trained_Model_Path>
--output_dir <Output_Dir>