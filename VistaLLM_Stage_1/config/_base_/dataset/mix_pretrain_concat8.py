_base_ = ['DEFAULT_TRAIN_DATASET.py']

data_args = dict(
    #
    train=dict(
        type='ConcatDataset',
        cfgs=[
            dict(
                type='SubSet',
                portion=1/20,
                do_shuffle=True,
                seed=42,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.gc}}, ### GC.json (similar to REG)
            ),
            dict(
                type='SubSet',
                portion=1/20,
                do_shuffle=True,
                seed=43,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.recvg}}, ### REC_ChatGPT.json 
            ),

            {{_base_.DEFAULT_TRAIN_DATASET.llavacc3m}}, ### Should be short
            {{_base_.DEFAULT_TRAIN_DATASET.llavalcs}}, ### Should be short

            {{_base_.DEFAULT_TRAIN_DATASET.VQAv2_train}}, ### VQA.json
            {{_base_.DEFAULT_TRAIN_DATASET.VQAE_train}}, ### VQA_CoT.json (May be long)
            {{_base_.DEFAULT_TRAIN_DATASET.VQAX_train}}, ### VQA_CoT.json (May be long)

            {{_base_.DEFAULT_TRAIN_DATASET.caption}}, ### image_cap.json

            {{_base_.DEFAULT_TRAIN_DATASET.rec}}, ### REC_ChatGPT.json 
            {{_base_.DEFAULT_TRAIN_DATASET.grec}}, ### GREC_ChatGPT.json 
            {{_base_.DEFAULT_TRAIN_DATASET.gres}}, ### GRES_ChatGPT.json
            {{_base_.DEFAULT_TRAIN_DATASET.reg}}, ### REG.json 
            {{_base_.DEFAULT_TRAIN_DATASET.res}}, ### RES_ChatGPT.json

            {{_base_.DEFAULT_TRAIN_DATASET.flickr}}, ### flicker30k_ChatGPT.json

            {{_base_.DEFAULT_TRAIN_DATASET.VCR_q_ra}}, ### VQA_BCoT_ChatGPT.json (May be long)
            {{_base_.DEFAULT_TRAIN_DATASET.VCR_qc_rac}}, ### VQA_BCoT_ChatGPT.json (May be long)

            {{_base_.DEFAULT_TRAIN_DATASET.VCR_qc_a}}, ### VQA.json
            {{_base_.DEFAULT_TRAIN_DATASET.VCR_qac_r}}, ### VQA.json
            # {{_base_.DEFAULT_TRAIN_DATASET.VCR_qc_a_qc_r}}, ### VQA_VCR_qc_a_qc_r.json

            {{_base_.DEFAULT_TRAIN_DATASET.POINT_LOCAL_b}}, ### All VQA.json
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_LOCAL_p}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_TWICE_oq_bp}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_TWICE_sq_bp}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_TWICE_gq_bp}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_V7W_p}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_V7W_b}},

        ],
    ),
    validation=None,
    test=None,

    # compute_metric
    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=1024,
        num_beams=1,
    ),
)
