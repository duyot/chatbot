Help me to planout the feature below:

1. create a select to list out all uploaded document 
2. create new python endpoint to accept the chat message. upon user choose the document, enter message and submit, it will call to this endpoint.
3. the endpoint will do the following
3.1. do the agentic rag to search from vector database, which the document saved before.
3.2. get the final response and stream to the front end.
4. after submit, front end will wait and receive the streaming response from api, then show in the ui.

Note: The item (3) is the most important when implement Agentic RAG.
Do some research for the Agentic RAG and propose best and most updated architecture
for it.