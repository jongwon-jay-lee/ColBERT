BSIZE=32

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
python -m torch.distributed.launch --nproc_per_node=8 -m \
colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize $BSIZE --accum 1 \
--triples /app/ColBERT/data/triples.train.small.tsv \
--root /app/ColBERT/outputs/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
