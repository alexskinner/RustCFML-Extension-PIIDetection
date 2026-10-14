---
library_name: transformers
datasets:
- nvidia/Nemotron-PII
language:
- en
base_model:
- jhu-clsp/ettin-encoder-17m
license: mit
metrics:
- f1
- precision
- recall
tags:
- pii-detection
- ner
- PII
- privacy
- token-classification
model-index:
  - name: ettin-17m-nemotron-pii
    results:
      - task:
          type: token-classification
          name: Named Entity Recognition
        dataset:
          name: nvidia/Nemotron-PII 
          type: nvidia/Nemotron-PII
          split: test (10K instances)
        metrics:
          - type: f1
            value: 0.9421
            name: F1 (micro)
          - type: precision
            value: 0.9448
            name: Precision
          - type: recall
            value: 0.9393
            name: Recall
---

# ettin-17m-nemotron-pii model

Light Weight PII Detection Model | Open Source | 17M Parameters | 94.21 F1 Score | [Blog Post](https://huggingface.co/blog/kalyan-ks/tiny-pii-entity-detection-models)

## Overview

Ettin-17m-nemotron-pii is based on the ettin-encoder-17M model and fine-tuned over the Nemotron PII dataset. This model can detect 50+ PII entities in both structured and unstructured texts across various domains like healthcare, finance, legal, cybersecurity etc. With just 17M parameters, the model achieves a strong F1-score of 94.21.

## Key Features 

- Achieves strong F1-score of 94.21 with just 17M parameters.
- Outperforms popular LLMs like DeepSeek-V4-Flash (84.89) and GPT-4o-Mini (78.69).
- Detects 50+ PII entities in both structured and unstructured texts.
- Handles text in various domains like healthcare, finance, legal etc.

## Supported PII Entity Types

This model can detect the following 55 PII entity types

<details>
<summary> PII entity types with description </summary>

| Entity                              | Description                                      |
|-------------------------------------|--------------------------------------------------|
| account_number                      | Account Number                                   |
| age                                 | Age                                              |
| api_key                             | API Key                                          |
| bank_routing_number                 | Bank Routing Number                              |
| biometric_identifier                | Biometric Identifier                             |
| blood_type                          | Blood Type                                       |
| certificate_license_number          | Certificate or License Number                    |
| city                                | City                                             |
| company_name                        | Company Name                                     |
| coordinate                          | Geographic Coordinate                            |
| country                             | Country                                          |
| county                              | County                                           |
| credit_debit_card                   | Credit or Debit Card Number                      |
| customer_id                         | Customer ID                                      |
| cvv                                 | Card Verification Value (CVV)                    |
| date                                | Date                                             |
| date_of_birth                       | Date of Birth                                    |
| date_time                           | Date and Time                                    |
| device_identifier                   | Device Identifier                                |
| education_level                     | Education Level                                  |
| email                               | Email Address                                    |
| employee_id                         | Employee ID                                      |
| employment_status                   | Employment Status                                |
| fax_number                          | Fax Number                                       |
| first_name                          | First Name                                       |
| gender                              | Gender                                           |
| health_plan_beneficiary_number      | Health Plan Beneficiary Number                   |
| http_cookie                         | HTTP Cookie                                      |
| ipv4                                | IPv4 Address                                     |
| ipv6                                | IPv6 Address                                     |
| language                            | Language                                         |
| last_name                           | Last Name                                        |
| license_plate                       | Vehicle License Plate                            |
| mac_address                         | MAC Address                                      |
| medical_record_number               | Medical Record Number                            |
| national_id                         | National Identification Number                   |
| occupation                          | Occupation                                       |
| password                            | Password                                         |
| phone_number                        | Phone Number                                     |
| pin                                 | Personal Identification Number (PIN)             |
| political_view                      | Political View                                   |
| postcode                            | Postcode / Zip Code                              |
| race_ethnicity                      | Race or Ethnicity                                |
| religious_belief                    | Religious Belief                                 |
| sexuality                           | Sexuality / Sexual Orientation                   |
| ssn                                 | Social Security Number                           |
| state                               | State                                            |
| street_address                      | Street Address                                   |
| swift_bic                           | SWIFT / BIC Code                                 |
| tax_id                              | Tax Identification Number                        |
| time                                | Time                                             |
| unique_id                           | Unique Identifier                                |
| url                                 | URL / Web Address                                |
| user_name                           | Username                                         |
| vehicle_identifier                  | Vehicle Identification Number (VIN)              |

</details>

## Usage

```python

# First install Hugging Face transformers library
!pip install transformers

# Initialize and run the PII detection pipeline to extract PII entities
from transformers import pipeline

## Initialize the PII detection pipeline
ner = pipeline("ner", model="kalyan-ks/ettin-17m-nemotron-pii", aggregation_strategy="simple")

input_text = "Kalyan KS is from India. His email id is kalyan.ks@yahoo.com"

## Run the PII detection to extract PII entities
pii_entities = ner(input_text)

## Process the extracted PII entities 
def format_pii_entities(entities, original_text):
    if not entities:
        return []

    merged_entities = []

    entities = sorted(entities, key=lambda x: x['start'])

    current_entity = {
        'start': entities[0]['start'],
        'end': entities[0]['end'],
        'label': entities[0]['entity_group'],
        'text': entities[0]['word']
    }

    for next_ent in entities[1:]:
        is_same_label = next_ent['entity_group'] == current_entity['label']
        is_adjacent = next_ent['start'] <= current_entity['end'] + 1

        if is_same_label and is_adjacent:
            current_entity['end'] = max(current_entity['end'], next_ent['end'])
            current_entity['text'] = original_text[current_entity['start']:current_entity['end']]
        else:
            merged_entities.append(clean_entity(current_entity))
            current_entity = {
                'start': next_ent['start'],
                'end': next_ent['end'],
                'label': next_ent['entity_group'],
                'text': next_ent['word']
            }

    merged_entities.append(clean_entity(current_entity))
    return merged_entities

def clean_entity(ent):

    raw_text = ent['text']
    stripped_text = raw_text.strip()
    leading_spaces = len(raw_text) - len(raw_text.lstrip())

    return {
        'start': ent['start'] + leading_spaces,
        'end': ent['start'] + leading_spaces + len(stripped_text),
        'text': stripped_text,
        'label': ent['label']
    }

# Display the extracted PII entities
formatted_entities = format_pii_entities(pii_entities, input_text)
print(formatted_entities)

# Output
[{'start': 0, 'end': 9, 'text': 'Kalyan KS', 'label': 'first_name'}, {'start': 18, 'end': 23, 'text': 'India', 'label': 'country'}, {'start': 41, 'end': 60, 'text': 'kalyan.ks@yahoo.com', 'label': 'email'}]
```


## Evaluation

This model is evaluated on a 10k sample test set from Neomotron PII dataset and achieved the following results

| Metric    | Score  |
|-----------|--------|
| F1        | 94.21  |
| Precision | 94.48  |
| Recall    | 93.93  |
| Accuracy  | 98.94  |

### Top Performing PII Entity Types

| Entity                              | Precision | Recall | F1     |
|-------------------------------------|----------|--------|--------|
| date_of_birth                       | 0.9915   | 0.9960 | 0.9938 |
| email                               | 0.9921   | 0.9926 | 0.9924 |
| biometric_identifier                | 0.9896   | 0.9951 | 0.9924 |
| employee_id                         | 0.9873   | 0.9918 | 0.9895 |
| vehicle_identifier                  | 0.9864   | 0.9904 | 0.9884 |
| mac_address                         | 0.9825   | 0.9929 | 0.9877 |
| ipv6                                | 0.9807   | 0.9946 | 0.9876 |
| health_plan_beneficiary_number      | 0.9953   | 0.9788 | 0.9869 |
| coordinate                          | 0.9766   | 0.9943 | 0.9854 |
| medical_record_number               | 0.9898   | 0.9799 | 0.9848 |

### Challenging PII Entity Types

| Entity           | Precision | Recall | F1     |
|------------------|----------|--------|--------|
| occupation       | 0.6747   | 0.4643 | 0.5500 |
| time             | 0.8499   | 0.7607 | 0.8028 |
| political_view   | 0.8202   | 0.8047 | 0.8124 |
| race_ethnicity   | 0.8170   | 0.8485 | 0.8324 |
| state            | 0.8550   | 0.8135 | 0.8337 |
| age              | 0.8307   | 0.8442 | 0.8374 |
| company_name     | 0.8386   | 0.8392 | 0.8389 |
| city             | 0.8514   | 0.8613 | 0.8563 |
| fax_number       | 0.8752   | 0.8406 | 0.8576 |
| national_id      | 0.8458   | 0.8716 | 0.8585 |

## Limitations

- **Language**: This model works well only for English language texts.
- **Challenging PII Entity Types**: Some of the entity types like `occupation` has low F1 score.

## Citation

```bibtex
@misc{ettin-17m-pii-2026,
  title = {ettin-17m-nemotron-pii-2026: PII Detection Model},
  author = {Kalyan KS},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/kalyan-ks/ettin-17m-nemotron-pii}
}
```