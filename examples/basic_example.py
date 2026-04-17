"""  Basic usage example of StudSar.
This script demonstrates how to use StudSar to create a neural network,
add text, perform queries, and save/load the state. """ 
import sys
import os

#You should add the main directory to the path here in order to import the StudSar package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

# Ora che il percorso è stato aggiunto, possiamo importare StudSarManager
from src.managers.manager import StudSarManager

def run_example():
    # EXAMPLE TEXT
    example_text = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "as opposed to the natural intelligence displayed by humans or animals. "
        "Leading AI textbooks define the field as the study of 'intelligent agents': "
        "any system that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. "
        # Add more text for better testing...
        "Some popular accounts use the term 'artificial intelligence' to describe machines that mimic 'cognitive' functions that humans associate with the human mind, "
        "such as 'learning' and 'problem-solving', however, this definition is rejected by major AI researchers. "
        "AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, generative tools, "
        "and competing at the highest level in strategic games. "
        "Machine learning is a core part of modern AI."
    )
    # CREATE MANAGER AND BUILD NETWORK
    studsar_manager = StudSarManager()
    #  New update for V2: Add a default emotion during construction.  
    studsar_manager.build_network_from_text(
        example_text,
        use_spacy_segmentation=False, # Disabilitato a causa di problemi di installazione di spaCy
        spacy_sentences_per_segment=2,
        default_emotion="neutral" # Aggiunta emozione di default
    )
    # END OF MODIFICATION V2 

    # EXECUTE A QUERY
    query = "What are AI applications?"
    ids, sims, segs = studsar_manager.search(query, k=2)
    if ids:
         print("Best results found for query:")
         for i in range(len(ids)):
              print(f"  {i+1}. ID: {ids[i]}, Sim: {sims[i]:.4f} - Seg: '{segs[i][:120]}...'")


    # UPDATE THE NETWORK
    new_text = "Deep learning is a subset of machine learning based on artificial neural networks with representation learning. It enables computers to learn from experience and understand the world in terms of a hierarchy of concepts."
    # EDIT V2: Add specific emotion when updating
    new_id = studsar_manager.update_network(new_text, emotion="informative")
    # END OF MODIFICATION V2 

    # RE-EXECUTE QUERY TO SEE IF NEW SEGMENT IS FOUND
    query_dl = "Tell me about deep learning"
    ids_dl, sims_dl, segs_dl = studsar_manager.search(query_dl, k=1)
    if ids_dl:
         print("Best result for 'deep learning':")
         print(f"  - ID: {ids_dl[0]}, Sim: {sims_dl[0]:.4f} - Seg: '{segs_dl[0][:120]}...'")
         # Check if ID matches the one just added
         if new_id is not None and ids_dl[0] == new_id:
              print("  (Correctly identified newly added segment!)")

    #  NEW V2 - Demonstrate V2 features 
    print("\n--- Demonstrating V2 Features ---")
    # Update reputation for the first found marker from the initial search
    if ids:
        marker_to_update = ids[0]
        print(f"Updating reputation for marker {marker_to_update}...")
        studsar_manager.update_marker_reputation(marker_to_update, 1.5) # Positive feedback
        studsar_manager.update_marker_reputation(marker_to_update, -0.5) # Negative feedback example 
        # Get and print details for the updated marker
        print(f"Retrieving details for marker {marker_to_update}...")
        details = studsar_manager.get_marker_details(marker_to_update)
        # Note: get_marker_details returns a dictionary: {"embedding": ..., "segment": ..., ...}
        if details: # Check if marker details dictionary was returned
            segment_text = details.get("segment", "N/A")
            emotion = details.get("emotion", "N/A")
            reputation = details.get("reputation", 0.0)
            usage = details.get("usage_count", 0)
            print(f"Details for marker {marker_to_update}:")
            print(f"  Segment: '{segment_text[:60]}...'")
            print(f"  Emotion: {emotion}")
            print(f"  Reputation: {reputation:.2f}")
            print(f"  Usage Count: {usage}") # Usage count incremented by search
        else:
            print(f"Could not retrieve details for marker {marker_to_update}.")

    # Get details for the newly added 'deep learning' marker
    if new_id is not None:
        print(f"\nRetrieving details for newly added marker {new_id}...")
        details_new = studsar_manager.get_marker_details(new_id)
        if details_new: # Check if marker details dictionary was returned
            segment_text_new = details_new.get("segment", "N/A")
            emotion_new = details_new.get("emotion", "N/A")
            reputation_new = details_new.get("reputation", 0.0)
            usage_new = details_new.get("usage_count", 0)
            print(f"Details for marker {new_id}:")
            print(f"  Segment: '{segment_text_new[:60]}...'")
            print(f"  Emotion: {emotion_new}") # Should be 'informative'
            print(f"  Reputation: {reputation_new:.2f}") # Should be 0.0 initially
            print(f"  Usage Count: {usage_new}") # Should be 1 after the search for 'deep learning'
        else:
            print(f"Could not retrieve details for marker {new_id}.")

    # Call placeholder methods (assuming they exist in manager, even if just printing)
    # studsar_manager.dream_mode_consolidation() # Uncomment if implemented
    studsar_manager.visualize_graph(output_file="studsar_example_graph.png") # Pass output file name
    # --- AN2 ---

    # 6. SAVE STATE
    save_path = "studsar_neural_demo.pth"
    studsar_manager.save(save_path)

    # 7. DELETE AND RELOAD
    del studsar_manager
    print("\nManager instance deleted. Attempting reload...")
    studsar_reloaded = StudSarManager.load(save_path)

    # 8. VERIFY LOADING AND RE-EXECUTE QUERY
    if studsar_reloaded:
        print("\nStudSar instance successfully reloaded!")
        query_post_load = "What is the definition of AI?"
        ids_post, sims_post, segs_post = studsar_reloaded.search(query_post_load, k=1)
        if ids_post:
             print("Best result (post-load):")
             print(f"  - ID: {ids_post[0]}, Sim: {sims_post[0]:.4f} - Seg: '{segs_post[0][:120]}...'")
             # --- NEWV2: Verify loaded V2 data ---
             print(f"    Verifying loaded V2 details for marker {ids_post[0]}...")
             details_post_load = studsar_reloaded.get_marker_details(ids_post[0])
             if details_post_load: # Check if marker details dictionary was returned
                 emotion_load = details_post_load.get("emotion", "N/A")
                 reputation_load = details_post_load.get("reputation", 0.0)
                 usage_load = details_post_load.get("usage_count", 0)
                 # Correctly escaped f-string
                 print(f"    (Loaded V2 Details: Rep={reputation_load:.2f}, Usage={usage_load}, Emotion={emotion_load})")
             else:
                 print(f"    Could not retrieve loaded details for marker {ids_post[0]}.")
             # --- AN2 ---
        else:
            print("No results found for query post-load.")
    else:
        print("Error during loading.")
    print("\n -- Example Complete --- 18/96¿ ")
if __name__ == "__main__":
    run_example()