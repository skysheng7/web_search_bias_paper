# Data and Code for Paper: "The Guardian is credible but leans towards opinion": Shadow mechanisms in GPT-5's web search and the politics of credibility

This repository contains the data, code, and analysis pipeline accompanying the paper
**_"The Guardian is credible but leans towards opinion": Shadow mechanisms in GPT-5's web search and the politics of credibility_**.

The project investigates how GPT-5's built-in web search behaves when asked politically charged questions about real-world politicians. We test two opposing framings of the same underlying question for 30 politicians from the **US**, **UK**, and **Europe**, and audit the model's internal reasoning, web search queries, retrieved sources, and final citations to understand how the model constructs and assigns "credibility" to news domains.

---

## Repository structure

```
web_search_bias/
├── 0_prompt_llms/                  # Data collection: prompting GPT-5 and logging responses
│   ├── code/                       # API clients, batch runners, log reconstruction
│   ├── data/                       # media bias / credibility ratings
│   └── result/                     # Raw GPT-5 responses (JSON) + human-readable logs (MD)
│       └── trial_1_reason_url/     # Main experimental run (US, UK, Europe)
│
├── 1_data_analysis/                # Analysis pipeline over the collected data
│   ├── code/                       # Domain processing, summary tables, plot generators
│   ├── manual_label/               # Hand-labeled web-search query domains by region
│   ├── results/                    # Processed CSVs (counts, summaries, citations)
│   └── plots/                      # Final figures used in the paper
│
├── environment.yml                 # Conda environment specification
└── README.md
```
