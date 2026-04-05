import spacy
from fastcoref import spacy_component

# 1. Base NLP purely for fast sentence splitting (No heavy processing)
nlp_base = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer"])
nlp_base.add_pipe("sentencizer")

# 2. Heavy NLP for Coreference and Dependency Parsing on the GPU
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("fastcoref", config={"model_architecture": "FCoref", "device": "cuda"})

def chunk_text(text, max_words=350):
    """
    Safely groups sentences into chunks under the transformer token limit
    without breaking sentence boundaries.
    """
    doc = nlp_base(text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in doc.sents:
        sent_len = len(sent.text.split())
        # If adding this sentence exceeds our safe limit, save the chunk and start a new one
        if current_len + sent_len > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent.text]
            current_len = sent_len
        else:
            current_chunk.append(sent.text)
            current_len += sent_len
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def extract_causal_triplets(scene_text):
    """
    Extracts complete (Subject, Predicate, Object) triplets for the entire scene.
    """
    chunks = chunk_text(scene_text)
    all_scene_triplets = []
    
    for chunk in chunks:
        # Process each safe-sized chunk on the GPU
        doc = nlp(chunk, component_cfg={"fastcoref": {"resolve_text": True}})
        resolved_text = doc._.resolved_text if doc._.resolved_text else chunk
        
        resolved_doc = nlp(resolved_text)
        
        # Extract triplets from the resolved chunk
        for sent in resolved_doc.sents:
            subj, obj, pred = None, None, None
            for token in sent:
                if token.pos_ == "VERB" and token.dep_ == "ROOT":
                    pred = token.lemma_
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            subj = child.text
                        if child.dep_ in ("dobj", "pobj", "attr"):
                            obj = child.text
                            
            if subj and pred and obj:
                all_scene_triplets.append((subj, pred, obj))
                
    return all_scene_triplets