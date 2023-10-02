from contrast_training_with_da import *
import os
import contextlib

source_domains = ["ctrl", "fair_wmt19", "gpt2_xl", "gpt3", "grover_mega", "xlm", "chatgpt"]
target_domains = ["ctrl", "fair_wmt19", "gpt2_xl", "gpt3", "grover_mega", "xlm", "chatgpt"]


loss_type = 'simclr'  

for src in source_domains:
    for tgt in target_domains:
        if src!=tgt:
            parser = argparse.ArgumentParser()

            parser.add_argument('--max-epochs', type=int, default=None)
            parser.add_argument('--device', type=str, default=None)
            parser.add_argument('--batch-size', type=int, default=16)
            parser.add_argument('--max-sequence-length', type=int, default=256)
            parser.add_argument('--random-sequence-length', action='store_true')
            parser.add_argument('--epoch-size', type=int, default=None)
            parser.add_argument('--loss_type', type=str, default=loss_type)  #ce_mmd (for loss 2), either of[simclr, supcon] (for loss 1)
            parser.add_argument('--seed', type=int, default=None)
            if src=='chatgpt':
                parser.add_argument('--src_data-dir', type=str, default='/home/abhatt43/projects/Data_for_Testing/syn_rep/ChatGPT/')
                parser.add_argument('--src_real-dataset', type=str, default=src+'_real')
                parser.add_argument('--src_fake-dataset', type=str, default=src+'_fake')
            else:
                parser.add_argument('--src_data-dir', type=str, default='/home/abhatt43/projects/Data_for_Testing/syn_rep/TuringBench/TT_'+src+'/')
                parser.add_argument('--src_real-dataset', type=str, default='tb_tt_'+src+'_real')
                parser.add_argument('--src_fake-dataset', type=str, default='tb_tt_'+src+'_fake')

            if tgt=='chatgpt':
                parser.add_argument('--tgt_data-dir', type=str, default='/home/abhatt43/projects/Data_for_Testing/syn_rep/ChatGPT/')
                parser.add_argument('--tgt_real-dataset', type=str, default=tgt+'_real')
                parser.add_argument('--tgt_fake-dataset', type=str, default=tgt+'_fake')

            else:
                parser.add_argument('--tgt_data-dir', type=str, default='/home/abhatt43/projects/Data_for_Testing/syn_rep/TuringBench/TT_'+tgt+'/')
                parser.add_argument('--tgt_real-dataset', type=str, default='tb_tt_'+tgt+'_real')
                parser.add_argument('--tgt_fake-dataset', type=str, default='tb_tt_'+tgt+'_fake')

            parser.add_argument('--model_save_path', default=os.getcwd()+'/models/')
            
            if loss_type=='simclr':
                parser.add_argument('--model_save_name', default=src+'_'+tgt+'_syn_rep_'+loss_type+'.pt')
            ## structure:  src-domain_tgt-domain_augmentation-type_loss-type.pt

            parser.add_argument('--token-dropout', type=float, default=None)

            parser.add_argument('--large', action='store_true', help='use the roberta-large model instead of roberta-base')
            parser.add_argument('--learning-rate', type=float, default=2e-5)
            parser.add_argument('--weight-decay', type=float, default=0)
            parser.add_argument('--load-decay', type=float, default=0)
            parser.add_argument('--lambda_w', type=float, default=0.5)

            args = parser.parse_args(args=['--max-epochs=5', '--model_save_path='+os.getcwd()+'/models/'])
            
            
            filename = 'output_log_'+src+'_'+tgt+'_syn_rep_'+loss_type+'.log'
            filepath = os.getcwd() + '/output_logs/'
            with open(filepath+filename, 'w') as f:
                with contextlib.redirect_stderr(f):
                    main(args)
