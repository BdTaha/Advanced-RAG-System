import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
from app.services.llm_service import generate_answer

context = "L'abeille est un insecte pollinisateur très important. Elle produit du miel et vit dans une ruche."
query = "Que produit l'abeille ?"

prompt1 = f"""En te basant uniquement sur le contexte fourni, réponds à la question de manière concise.
Si la réponse n'est pas dans le contexte, dis 'Information non trouvée'.

Question: {query}

Contexte:
{context}

Réponse:"""

prompt2 = f"""Answer the following question based only on the provided context. Answer in French.

Context:
{context}

Question: {query}

Answer:"""

print('PROMPT 1:')
print(generate_answer(prompt1))
print('PROMPT 2:')
print(generate_answer(prompt2))
