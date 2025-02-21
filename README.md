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

## Key Components

### Core Modules
- `core/gemini_analyzer.py`: AI-powered transaction parsing and analysis
- `core/ynab_client.py`: YNAB API interaction
- `core/shared_models.py`: Data models and validation

### Key Capabilities
- Robust amount normalization
- Comprehensive error handling
- Context-aware transaction processing
- Semantic matching for categorization

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
python -m cli.natural_language_cli process "Create a transaction for $50 at Target"

# Streamlit Web Interface
streamlit run app.py
```

## Development Principles

- 🔒 Comprehensive Error Handling
- 🧩 Dependency Injection
- 🔍 Semantic Matching
- 📈 Confidence Scoring

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