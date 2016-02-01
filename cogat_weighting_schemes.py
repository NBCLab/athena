# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:37:57 2016
Weighting schemes for CogAt
@author: salo
"""

def get_weights(scheme):
    schemes = {"none": {"isSelf": 1},
               "ws1": {"isSelf": 1,
                       "isKindOf": 1,
                       "inCategory": 1,
                       "childOf": 1,
                       },
               "ws2": {"isSelf": 1,
                       "isKindOf": 1,
                       "isPartOf": {"num": 0.5,
                                    "den": "n"},
                       "hasKind": {"num": 1,
                                   "den": "n"},
                       "hasPart": {"num": 0.25,
                                   "den": "n"},
                       "relatedTasks": {"num": 1,
                                        "den": "n"},
                       "inCategory": 1,
                       "descendantOf": 0,
                       "progenitorOf": 0,
                       "relatedConcepts": {"num": 1,
                                           "den": "n"},
                       "relatedDisorders": 0,
                       "childOf": 1,
                       "parentOf": {"num": 1,
                                    "den": "n"},
                       "relatesTasks2": 0,
                       },
               }
    return schemes[scheme]