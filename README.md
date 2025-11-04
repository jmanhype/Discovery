# Discovery

This is an AI agent simulation project that demonstrates the collaboration, interactions, and learning of multiple agents in a shared environment. Each AI agent is equipped with a Theory of Mind, natural language understanding, and probabilistic reasoning capabilities. The project also supports a visual user interface for user interaction and monitoring.

## Features

- Multiple AI agents with diverse personalities
- Natural Language Processing for communication
- Theory of Mind for understanding and interpreting other agents' actions
- Probabilistic reasoning module for evaluating and updating beliefs
- Active learning and probabilistic programming
- Real-time decision-making based on obtained beliefs
- User interface for real-time interaction and visualization
- Performance metrics for evaluating agents' understanding capabilities

## Modules

##### LanguageUnderstandingModule
- Translation of text to PLoT (Probabilistic Logical Theories) expressions
- Loading and training of custom models for specific languages

##### ProbabilisticReasoningModule
- Evaluation of PLoT expressions
- Updating beliefs based on observations and evidence

##### TheoryOfMind
- Interpretation of other agents' actions to update beliefs and confidence
- Querying other agents to gather more information

##### Agent
- Responds to the world state and the actions of other agents
- Makes decisions, updates beliefs, and takes actions based on reasoning and learning

##### Simulation
- World state management and agent initiation
- Ticking the simulation to cycle through actions
- GUI interface for user monitoring and interaction

## How to run

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key as an environment variable:

   **Linux/macOS:**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

   **Windows (Command Prompt):**
   ```cmd
   set OPENAI_API_KEY=your-api-key-here
   ```

   **Windows (PowerShell):**
   ```powershell
   $env:OPENAI_API_KEY='your-api-key-here'
   ```

   Alternatively, create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. Run the `main.py` script:

   ```bash
   python main.py
   ```

This will initialize the simulation and run it until the termination condition is met. The interface will display the state of the simulation and agent interactions in real time.

## Security Note

**Important:** Never commit your API keys to version control. Always use environment variables or a `.env` file (which should be added to `.gitignore`).
