import os
import time
import faiss
import random
import torch
import itertools

from colbert.reading.extractive_reader import load_extractive_reader, extractive_answer
from colbert.utils.runs import Run
from multiprocessing import Pool
from colbert.modeling.inference import ModelInference
from colbert.evaluation.ranking_logger import RankingLogger

from colbert.utils.utils import print_message, batch
from colbert.ranking.rankers import Ranker


def read(args):
    # load colbert model and ready query/passage tokenizers
    inference = ModelInference(args.colbert, amp=args.amp)
    ranker = Ranker(args, inference, faiss_depth=args.faiss_depth)

    ranking_logger = RankingLogger(Run.path, qrels=None)
    milliseconds = 0

    with ranking_logger.context('ranking.tsv', also_save_annotations=False) as rlogger:
        queries = args.queries
        qids_in_order = list(queries.keys())

        for qoffset, qbatch in batch(qids_in_order, 100, provide_offset=True):
            qbatch_text = [queries[qid] for qid in qbatch]

            rankings = []
            rankings_for_qa = []

            for query_idx, q in enumerate(qbatch_text):
                torch.cuda.synchronize('cuda:0')
                s = time.time()

                Q = ranker.encode([q])
                pids, scores = ranker.rank(Q)

                torch.cuda.synchronize()
                milliseconds += (time.time() - s) * 1000.0

                if len(pids):
                    print(qoffset+query_idx, q, len(scores), len(pids), scores[0], pids[0],
                          milliseconds / (qoffset+query_idx+1), 'ms')

                # rankings.append(zip(pids, scores))
                rank_zip = zip(pids, scores)
                ranking, ranking_qa = itertools.tee(rank_zip)
                rankings.append(ranking)
                rankings_for_qa.append(ranking_qa)

            for query_idx, (qid, ranking) in enumerate(zip(qbatch, rankings)):
                query_idx = qoffset + query_idx

                if query_idx % 100 == 0:
                    print_message(f"#> Logging query #{query_idx} (qid {qid}) now...")

                ranking = [(score, pid, None) for pid, score in itertools.islice(ranking, args.depth)]
                rlogger.log(qid, ranking, is_ranked=True)

            # reader
            reader_model, reader_tokenizer = load_model(args)
            for i, (qid, ranking_qa) in enumerate(zip(qbatch, rankings_for_qa)):
                for j, (pid, score) in enumerate(itertools.islice(ranking_qa, args.top_n_psg)):
                    if qid in args.queries and str(pid) in args.collection_dict:
                        question = args.queries[qid]
                        passage = args.collection_dict[str(pid)]
                        # print(i, score, pid, passage)
                        answer, answer_start, answer_start_prob, answer_end, answer_end_prob = extractive_answer(question, passage, reader_model, reader_tokenizer)
                        print(f"\nRANK: {j}")
                        print(f"Question: {question}")
                        print(f"Context: {passage}")
                        print(f"Answer: {answer}")
                        print()


    print('\n\n')
    print(ranking_logger.filename)
    print("#> Done.")
    print('\n\n')
