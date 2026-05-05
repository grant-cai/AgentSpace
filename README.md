# How to use Agent Space

## Environment Set Up

1. If you do not have Conda download it

2. Create a new conda environment in terminal using:
```
conda create -n <environment_name> python=3.13

```

3. Activate the conda environment using:
```
conda activate <environment_name>
```

4. Download dependencies by running the following command in the same directory as `environment.yml`. If on apple use `environment.yml` on windows use `environment_windows.yml`:
```
conda env update -f environment.yml
```

## Using AgentSpace with Jupyter Notebooks

1. Navigate to the `KnowledgeRetriever` folder and open `Neo4jGraph.ipynb`

2. Follow the instructions in `Neo4jGraph.ipynb` to get Neo4j set up and create your graph

   > **Note:** These two variables are set:
   > ```
   > INDEX_NAME = "essay_chunk_agentspace"
   > CHUNK_NAME = "chunk_index"
   > ```
   > What they are set to must be passed into the agent. It is how the agent determines what documents to look at.
   > ```python
   > agent = personAgent("essay_chunk_agentspace", "chunk_index")
   > ```

3. Navigate to the `Agent` folder and open `TestAgent.ipynb`

4. Run `testAgent` to ask the agent questions

   > **Note:** Agent must be made using the `INDEX_NAME` and `CHUNK_NAME` set in `Neo4jGraph.ipynb`

## Current Repository Setup

```
├── Orchestrator/                 # AgentSpace Agent
│   ├── Agent.py                  # Defines the Agent, uses Knowledge and Personality Retrievers
│   └── TestAgent.ipynb           # Test an Agent
├── KnowledgeRetriever/           # Graph/retrievers for knowledge
│   ├── Retriever.py              # Holds the graph retriever
│   ├── Neo4jAdapter.py           # Wraps Retriever.py for use with Neo4j
│   └── Neo4jGraph.ipynb          # Jupyter notebook to set up Neo4j graph
├── PersonalityRetriever/         # Database/retrievers for personality
│   ├── faiss_db                  # faiss vector store
|   ├── create_profile.py         # Uses LLM API call to generate personality.json
|   ├── grant_tutor_rag           # Current personality pipeline. Creates faiss db
|   ├── personality_summary.json  # JSON profile. Serves as a general lookup of personality traits and characteristics.
|   └── personality_wrapper.py    # Wraps retrieval and loading of personality JSON
└── README.md                     # Project documentation
```

