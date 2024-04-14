#!/usr/bin/env python3
# CLMS Capstone | Dean Cahill 
# Clustering Preprocess Code --- Testing
# ============================================================|
# Import Statements
import pytest 
from src.clustering import clustering
from time import perf_counter
import os 

# Constants
FULL_DATASET = os.Path("/home/mr-squally/projects/brandeis_work/CLMS_Capstone/codebase/preprocessing-clustering/data/.jsonl")
TOY_DATASET = os.Path("test_data/toy_sample.jsonl")
EMPTY_DATASET = os.Path("test_data/empty.jsonl")
# ============================================================|
class TestClustering:
    
    gold_document_fields = ("document_id", "document_url", "question_text", "")
    test_docs: dict = {"first":"toy_sample.jsonl", 
                       "second":"empty.jsonl"}


    def test_load_data(self):
        """
        ## --- Test Cases --- 
        (1.) document is parsed as a Python dictionary
        (2.) top level document representation is reflective of the NQ document structure
        (3.) empty file doesn't throw an exception
        ## --- --- --- --- ---
        """

        # First Test - Case 1 & 2
        fileloader_one = clustering.load_data(TOY_DATASET)    
        sample_document = next(fileloader_one)
        document_fields = sample_document.keys()
        assert isinstance(sample_document, dict)
        assert document_fields == self.document_fields
        fileloader_one.close()

        # Second Test - Case 
        fileloader_two = clustering.load_data(EMPTY_DATASET)ff=
        assert not next(fileloader_two)

        
    def test_process_doc(self):
        pass 


class TestClusteringPerformance:
    def test_load_data_perf(self):
        """
        ## --- Test Cases ---
        (1.) generator is memory efficient             
        ## --- --- --- --- ---
        """
        fileloader = clustering.load_data(FULL_DATASET)        
        def dummy_fn(x: dict):
            y = x 
            return y
    
        for doc in fileloader:
            dummy_fn(doc)
        assert True 

            
        
                    

                

        
