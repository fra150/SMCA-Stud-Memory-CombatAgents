import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict # Import defaultdict
import time

class StudSarNeural(nn.Module):
    """
    Core neural network for StudSar associative memory.
    Stores text segments and their embeddings (markers), enabling similarity search.
    """
    def __init__(self, embedding_dim, initial_capacity=1024, device=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"StudSarNeural initialized on device: {self.device}")

        # Use register_buffer for tensors that are part of the state but not model parameters
        self.register_buffer('memory_embeddings', torch.zeros(initial_capacity, self.embedding_dim, device=self.device))

        # Mappings stored as regular attributes (dictionaries are not parameters/buffers)
        self.id_to_segment = {}
        self.next_id = 0
        self.marker_id_to_index = {} # Map marker ID to tensor index

        # --- NEWV2 ---
        self.id_to_emotion = {} # Stores emotion tag per marker ID
        self.id_to_reputation = defaultdict(float) # Stores reputation score per marker ID, default 0.0
        self.id_to_usage = defaultdict(int) # Stores usage count per marker ID, default 0
        self.id_to_created_ts = {} # Creation timestamp per marker
        self.id_to_last_access_ts = {} # Last retrieval timestamp per marker
        self.id_to_decay_immune = defaultdict(bool) # Resurrection immunity flag
        # --- AN2 ---

        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Initial capacity: {initial_capacity} markers")

    def _ensure_capacity(self, required_index):
        """Dynamically increases memory capacity if needed."""
        current_capacity = self.memory_embeddings.shape[0]
        if required_index >= current_capacity:
            new_capacity = max(current_capacity * 2, required_index + 1)
            print(f"    Resizing memory_embeddings from {current_capacity} to {new_capacity}")
            new_embeddings = torch.zeros(new_capacity, self.embedding_dim, device=self.device)
            new_embeddings[:current_capacity] = self.memory_embeddings
            # We need to re-assign to update the buffer correctly
            # del self.memory_embeddings # Deregister old buffer (optional, PyTorch might handle this)
            self.register_buffer('memory_embeddings', new_embeddings) # Register new buffer

    #  EDIT V2: Added emotion parameter 
    def add_marker(self, segment_text, embedding, emotion=None):
        """Adds a new marker (segment + embedding) to the memory."""
        if not segment_text or embedding is None:
            print("Error: Cannot add marker with empty segment or None embedding.")
            return None

        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).to(self.device)
        elif not isinstance(embedding, torch.Tensor):
             print(f"Error: Embedding must be a numpy array or torch tensor, got {type(embedding)}")
             return None

        embedding = embedding.to(self.device) # Ensure embedding is on the correct device

        if embedding.shape[0] != self.embedding_dim:
             print(f"Error: Embedding dimension mismatch. Expected {self.embedding_dim}, got {embedding.shape[0]}")
             return None

        marker_id = self.next_id
        tensor_index = len(self.marker_id_to_index) # Next available index
        self._ensure_capacity(tensor_index) # Check capacity before adding
        self.memory_embeddings[tensor_index] = embedding.float() # Store as float
        self.id_to_segment[marker_id] = segment_text
        self.marker_id_to_index[marker_id] = tensor_index
        now = time.time()
        self.id_to_created_ts[marker_id] = now
        self.id_to_last_access_ts[marker_id] = now

        #  NEW ADDITIONS V2  
        if emotion:
            self.id_to_emotion[marker_id] = emotion
        # Reputation and Usage will use defaultdict defaults (0.0 and 0)
        # self.id_to_reputation[marker_id] = 0.0 # Explicitly set if not using defaultdict
        # self.id_to_usage[marker_id] = 0 # Explicitly set if not using defaultdict
        # e.END OF ADDITIONS V2  

        self.next_id += 1
        # print(f"  Marker added: ID={marker_id}, Index={tensor_index}, Emotion='{emotion if emotion else 'N/A'}'")
        return marker_id

    def get_total_markers(self):
        """Returns the current number of markers stored."""
        return len(self.marker_id_to_index)

    def search_similar_markers(self, query_embedding, k=1):
        """Finds the top k most similar markers to the query embedding."""
        num_markers = self.get_total_markers()
        if num_markers == 0:
            return [], [], []

        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding).to(self.device)
        elif not isinstance(query_embedding, torch.Tensor):
             print(f"Error: Query embedding must be a numpy array or torch tensor, got {type(query_embedding)}")
             return [], [], []

        query_embedding = query_embedding.to(self.device).float() # Ensure float and device

        # Calculate cosine similarity
        # Use only the populated part of the memory_embeddings tensor
        active_embeddings = self.memory_embeddings[:num_markers]
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), active_embeddings)

        #  MODIFICATION V3: Reputation-based search enhancement
        # Apply reputation boost to similarity scores
        index_to_marker_id = {v: k for k, v in self.marker_id_to_index.items()}
        reputation_boosts = torch.zeros(num_markers, device=self.device)
        
        for idx in range(num_markers):
            if idx in index_to_marker_id:
                marker_id = index_to_marker_id[idx]
                reputation = self.id_to_reputation.get(marker_id, 0.0)
                # Apply tanh-based boost: higher reputation = higher boost
                # Boost factor ranges from 0.9 to 1.1 based on reputation (-1.0 to 1.0)
                boost = 1.0 + 0.1 * torch.tanh(torch.tensor(reputation * 0.5, device=self.device))
                reputation_boosts[idx] = boost
        
        # Apply reputation boost to similarities
        similarities = similarities * reputation_boosts
        # END OF MODIFICATION V3

        # Get top k results
        k = min(k, num_markers) # Adjust k if fewer markers than requested
        top_k_similarities, top_k_indices_tensor = torch.topk(similarities, k)

        # Convert tensor indices back to marker IDs and get segments
        top_k_indices = top_k_indices_tensor.cpu().numpy()
        top_k_similarities = top_k_similarities.cpu().numpy()

        # Need to map tensor indices back to marker IDs
        index_to_marker_id = {v: k for k, v in self.marker_id_to_index.items()}
        result_ids = [index_to_marker_id[idx] for idx in top_k_indices if idx in index_to_marker_id]
        result_segments = [self.id_to_segment[m_id] for m_id in result_ids]
        result_similarities = [sim for idx, sim in zip(top_k_indices, top_k_similarities) if idx in index_to_marker_id]
        return result_ids, result_similarities, result_segments

    #  NEW ADDITION V2: Increase usage count
    def increment_usage(self, marker_id):
        """Increments the usage count for a given marker ID."""
        if marker_id in self.marker_id_to_index:
            self.id_to_usage[marker_id] += 1
            self.id_to_last_access_ts[marker_id] = time.time()
            # print(f"  Usage count for marker ID {marker_id} incremented to {self.id_to_usage[marker_id]}.") # Optional: uncomment for verbose logging
            return True
        else:
            # print(f"  Warning: Marker ID {marker_id} not found for usage increment.") # Optional: uncomment for verbose logging
            return False
    #  END OF MODIFICATION 

    #  NEW ADDITION V2   
    def update_marker_reputation(self, marker_id, feedback_score):
        """Updates the reputation score for a given marker ID."""
        if marker_id in self.marker_id_to_index:
            self.id_to_reputation[marker_id] += feedback_score
            self.id_to_reputation[marker_id] = max(-0.3, float(self.id_to_reputation[marker_id]))
            print(f"  Updated reputation for marker ID {marker_id}. New score: {self.id_to_reputation[marker_id]:.2f}")
            return True
        else:
            print(f"  Error: Marker ID {marker_id} not found for reputation update.")
            return False
    #  END OF MODIFICATION V2 

    #  NEW ADDITION V2 
    def get_marker_by_id(self, marker_id):
         """Retrieves all details for a specific marker ID as a dictionary."""
         if marker_id in self.marker_id_to_index:
              tensor_index = self.marker_id_to_index[marker_id]
              # Return embedding as numpy array for easier handling outside torch environment
              embedding = self.memory_embeddings[tensor_index].detach().cpu().numpy()
              segment = self.id_to_segment.get(marker_id, "Segment not found")
              emotion = self.id_to_emotion.get(marker_id) # Can be None
              reputation = self.id_to_reputation[marker_id] # Uses defaultdict
              usage = self.id_to_usage[marker_id] # Uses defaultdict
              return {
                  "embedding": embedding,
                  "segment": segment,
                  "emotion": emotion,
                  "reputation": reputation,
                  "usage_count": usage # Use 'usage_count' for clarity as used in example
              }
         else:
              # Return None or an empty dict to indicate not found
              return None
    #  END OF MODIFICATION V2

    # NEW ADDITION V2: Hook for View   
    def get_all_embeddings_and_ids(self):
        """Returns all active embeddings and their corresponding marker IDs."""
        num_markers = self.get_total_markers()
        if num_markers == 0:
            return {}, None

        active_embeddings = self.memory_embeddings[:num_markers].cpu() # Get active embeddings on CPU
        index_to_marker_id = {v: k for k, v in self.marker_id_to_index.items()}
        # Create a dictionary mapping marker_id to its embedding tensor
        id_to_embedding = {index_to_marker_id[i]: active_embeddings[i] for i in range(num_markers) if i in index_to_marker_id}
        return id_to_embedding
    #  END OF ADDITION V2   

    #  NEW ADDITION V2: Hook per Dream Mode 
    def update_marker_embedding(self, marker_id, new_embedding):
        """Updates the embedding for an existing marker."""
        if marker_id in self.marker_id_to_index:
            if isinstance(new_embedding, np.ndarray):
                new_embedding = torch.from_numpy(new_embedding).to(self.device)
            elif not isinstance(new_embedding, torch.Tensor):
                 print(f"Error: New embedding must be a numpy array or torch tensor.")
                 return False
            new_embedding = new_embedding.to(self.device).float()
            if new_embedding.shape[0] != self.embedding_dim:
                 print(f"Error: New embedding dimension mismatch.")
                 return False
            tensor_index = self.marker_id_to_index[marker_id]
            self.memory_embeddings[tensor_index] = new_embedding
            print(f"  Updated embedding for marker ID {marker_id}.")
            # Reset usage/reputation? Or keep? Current: Keep.
            # self.id_to_usage[marker_id] = 0 # Optional: Reset usage after update
            return True
        else:
            print(f"  Error: Marker ID {marker_id} not found for embedding update.")
            return False
    #  END OF ADDITION V2

    # TMDR: Temporal Memory Decay with Reinforcement Resurrection
    def apply_temporal_decay(self, decay_lambda: float = 0.0005) -> None:
        """
        Apply exponential decay to marker reputations over elapsed time since last access.
        decay_lambda is per-second rate; reputation is multiplied by exp(-lambda * dt).
        Markers with decay immunity are not decayed.
        """
        now = time.time()
        for mid in list(self.marker_id_to_index.keys()):
            if self.id_to_decay_immune.get(mid, False):
                continue
            last = float(self.id_to_last_access_ts.get(mid, self.id_to_created_ts.get(mid, now)))
            dt = max(now - last, 0.0)
            if dt <= 0.0:
                continue
            rep = float(self.id_to_reputation.get(mid, 0.0))
            if rep == 0.0:
                # even zero stays zero
                continue
            decayed = rep * float(np.exp(-decay_lambda * dt))
            self.id_to_reputation[mid] = decayed

    def grant_resurrection(self, marker_id: int) -> None:
        """Grant permanent decay immunity for a marker (God Protocol resolution)."""
        self.id_to_decay_immune[marker_id] = True

    def count_active_markers(self, threshold: float = 0.05) -> int:
        """
        Count markers considered 'active' under TMDR: decay-immune OR |reputation| >= threshold.
        """
        active = 0
        for mid in self.marker_id_to_index.keys():
            if self.id_to_decay_immune.get(mid, False):
                active += 1
            else:
                if abs(float(self.id_to_reputation.get(mid, 0.0))) >= threshold:
                    active += 1
        return active

    def forward(self, x):
        # This network doesn't have a traditional forward pass for training like classification models.
        # Its primary operations are add_marker and search_similar_markers.
        pass
