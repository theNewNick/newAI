# modules/system1/analysis_storage.py

from flask import session

###############################################################################
# GLOBAL IN-MEMORY STORE
###############################################################################
# NOTE: In a real production environment, you would NOT rely on this for
# permanent storage—if your server restarts, you lose all data. A better approach
# is to use a database or a cache (like Redis) for persistence and concurrency.
###############################################################################
global_analysis_results = {}

###############################################################################
# HELPER FUNCTIONS TO STORE / RETRIEVE RESULTS BY USER (OR SESSION, ETC.)
###############################################################################

def store_results_for_user(user_id, results_dict):
    """
    Stores the results of your analysis in the global dictionary under a specific
    user_id or session key.

    :param user_id: A string or unique identifier for the user/session.
    :param results_dict: A dictionary containing all analysis results.
    """
    global global_analysis_results
    global_analysis_results[user_id] = results_dict


def get_results_for_user(user_id):
    """
    Retrieves the results for the given user_id. Returns None if no data is found.

    :param user_id: The same string or identifier used when storing the results.
    :return: The dictionary of analysis results, or None if not found.
    """
    return global_analysis_results.get(user_id)
