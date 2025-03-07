# YNAB Streamlit AI Assistant

## Overview

A sophisticated financial assistant that leverages AI and the YNAB (You Need A Budget) API to provide intelligent transaction management, categorization, and insights.

## Features

- 🤖 Natural Language Transaction Creation
- 📊 Intelligent Transaction Categorization
- 🧠 AI-Powered Spending Analysis
- 💬 Conversational Financial Interface

## Technology Stack

- **Language**: Python 3.12
- **AI**: Google Gemini Pro
- **API Integration**: YNAB API
- **Web Framework**: Streamlit
- **Data Validation**: Pydantic
- **Logging**: Structured Logging

## Architecture

The application follows a modular architecture with strict separation of concerns:

### Core Layer
- **API**: Base and specialized API clients for YNAB
- **Services**: Business logic for transactions, categories, and analysis
- **Models**: Data structures and validation
- **Utils**: Reusable utilities for caching, error handling, etc.
- **Prompts**: AI prompt management and examples

### Interface Layer
- **CLI**: Command-line interface for script-based interactions
- **Streamlit**: Web interface for interactive usage

## Installation

```bash
# Clone the repository
git clone https://github.com/Meh-S-Eze/ynab-streamlit-agent.git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file with the following:
```
YNAB_API_KEY=your_ynab_api_key
YNAB_BUDGET_ID=your_budget_id
GEMINI_API_KEY=your_gemini_api_key
```

## Running the Application

```bash
# CLI Interface
./run_cli.py analyze-budget  # List budgets or analyze specific budget

# Natural Language CLI
./run_nl_cli.py "Show my grocery spending for last month"  # Process natural language query

# Streamlit Web Interface
./run_streamlit.py  # Launch the web interface
```

## Development Principles

- 🔒 Comprehensive Error Handling
- 🧩 Dependency Injection
- 🔍 Semantic Matching
- 📈 Confidence Scoring
- 🔄 Clean Architecture

## Directory Structure

```
src/
├── core/                      # Core business logic
│   ├── api/                   # YNAB API clients
│   ├── services/              # Business services
│   ├── models/                # Data models
│   ├── utils/                 # Utilities
│   ├── prompts/               # AI prompts
│   └── container.py           # Dependency injection container
├── cli/                       # CLI interface
│   ├── main.py                # Main CLI commands
│   └── natural_language_cli.py # Natural language CLI
└── streamlit/                 # Web interface
    └── app.py                 # Streamlit application
```

## Roadmap

- [ ] Multi-model AI Support
- [ ] Enhanced Spending Insights
- [ ] Machine Learning Categorization Improvements
- [ ] Advanced Budgeting Recommendations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Project Link: [https://github.com/Meh-S-Eze/ynab-streamlit-agent](https://github.com/Meh-S-Eze/ynab-streamlit-agent) 