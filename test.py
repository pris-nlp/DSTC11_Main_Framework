a = {
    'datasets': ['development'],
    'experiments': [
        {
        'dialogue_reader': 'default_dialogue_reader',
        'dialogues_path': 'dialogues.jsonl',
        'intent_clustering_model': {
            'clustering_algorithm': {
                'clustering_algorithm': {
                    'clustering_algorithm_name': 'kmeans',
                    'clustering_algorithm_params': {'n_init': 10},
                    'type': 'sklearn_clustering_algorithm'
                },
                'max_clusters': 50,
                'max_evals': 100,
                'metric': {
                    'metric_name': 'silhouette_score',
                    'metric_params': {'metric': 'cosine'},
                    'type': 'sklearn_clustering_metric'
                },
                'min_clusters': 5,
                'parameter_search_space': {
                    'n_clusters': ['quniform', 5, 50, 1]
                },
                'patience': 25,
                'tpe_startup_jobs': 10,
                'trials_per_eval': 3,
                'type': 'hyperopt_tuned_clustering_algorithm'
            },
            'embedding_model': {
                'model_name_or_path': 'sentence-transformers/average_word_embeddings_glove.840B.300d',
                'type': 'sentence_transformers_model'
            },
            'type': 'baseline_intent_clustering_model'
        },
        'run_id': 'kmeans_glove-840b-300d',
        'type': 'intent_clustering_experiment'
        },





        {'dialogue_reader': 'default_dialogue_reader',
         'dialogues_path': 'dialogues.jsonl',
         'intent_clustering_model': {
             'clustering_algorithm': {
                 'clustering_algorithm': {
                     'clustering_algorithm_name': 'kmeans',
                     'clustering_algorithm_params': {'n_init': 10},
                     'type': 'sklearn_clustering_algorithm'},
                 'max_clusters': 50,
                 'max_evals': 100,
                 'metric': {
                     'metric_name': 'silhouette_score',
                     'metric_params': {
                         'metric': 'cosine'
                     },
                     'type': 'sklearn_clustering_metric'
                 },
                 'min_clusters': 5,
                 'parameter_search_space': {
                     'n_clusters': ['quniform', 5, 50, 1]
                 },
                 'patience': 25,
                 'tpe_startup_jobs': 10,
                 'trials_per_eval': 3,
                 'type': 'hyperopt_tuned_clustering_algorithm'
             }, 'embedding_model': {
                 'cache_path': 'cache',
                 'prefix': 'all-mpnet-base-v2',
                 'sentence_embedding_model': {
                     'model_name_or_path': 'sentence-transformers/all-mpnet-base-v2',
                     'type': 'sentence_transformers_model'
                 },
                 'type': 'caching_sentence_embedding_model'
             }, 'type': 'baseline_intent_clustering_model'
         },
         'run_id': 'kmeans_all-mpnet-base-v2',
         'type': 'intent_clustering_experiment'
         }
        ],
    'run_id': 'intent-clustering-baselines',
    'type': 'meta_experiment'
}