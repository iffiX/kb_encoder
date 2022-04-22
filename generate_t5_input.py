import os
import json
import pprint
from encoder.dataset.arc import ARCDataset
from encoder.utils.settings import preprocess_cache_dir

if __name__ == "__main__":
    dataset = ARCDataset(
        None,
        matcher_mode="embedding",
        matcher_seed=697474,
        matcher_config={
            "question_match_max_times": 1000,
            "question_match_max_depth": 1,
            "question_match_edge_top_k": 10,
            "question_match_source_context_range": 0,
            "question_select_max_edges": 2,
            "question_select_discard_edges_if_rank_below": 0.4,
            "choices_match_max_times": 1000,
            "choices_match_max_depth": 1,
            "choices_match_edge_top_k": 10,
            "choices_match_source_context_range": 0,
            "choices_select_max_edges": 3,
            "choices_select_discard_edges_if_rank_below": 0.4,
        },
    )
    # with open(os.path.join(preprocess_cache_dir, "arc_targets.json"), "r") as file:
    #     dataset.set_search_targets(json.load(file))
    # pprint.pprint(dataset.generate_t5_annotation(dataset.test_data[0]))
    dataset.generate_all_t5_data()

# import os
# import json
# from encoder.dataset.matcher.fact_filter import FactFilter
# from encoder.utils.settings import preprocess_cache_dir
#
# if __name__ == "__main__":
#     with open(os.path.join(preprocess_cache_dir, "arc_facts.json"), "r") as file:
#         facts = json.load(file)
#     with open(
#         os.path.join(preprocess_cache_dir, "arc_filtered_facts.json"), "w"
#     ) as file:
#         json.dump(FactFilter().clean(facts), file, indent=2)
