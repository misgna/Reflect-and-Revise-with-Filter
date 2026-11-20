def get_rubric_r2c():
    return f"""
        # Content
        ## Novelty of ideas and content
        - The idea is ingenious.
        - The argument or evidence is fresh and novel.
        
        ## Richness of content
        - The content is varied, including diverse evidence and examples
        - The content is specific.
        
        ## Logicality of content
        - The evidence is valid and reasonable.
        - The author responds appropriately, taking into account expected counter arguments.
        
        # Organization
        ## Orginality of structure
        - The narative approach is unique.
        - The introduction and conclusion have been impressively structured.
        
        ## Cohesiveness of structure
        - Paragraphs are well separated, and the structure is systematic.
        - The text flows smoothly and is cohesive.
        
        # Expression
        ## Originality of expression
        - The expression is original and not clicheed, featuring creative metaphors, witty quotations, and literary expressions.
        
        ## Appropriateness of expression
        - The author uses accurate and objective words that fit the grammar.
        
        # Author's voice
        ## Perspective and personality
        - The author's new perspective on the topic is revealed.
        - The author's personality is revealed through his/her writing style and other elements.
        
        # Readers' response
        ## Fun and persuasiveness
        - The writing is fun and interesting.
        - The author's argument is persuasive and the reader is impressed.
    
    Based on the above criteria, classify the essay into creative, conventional, non-creative
    """, "Classify as creative, conventional and non-creative."