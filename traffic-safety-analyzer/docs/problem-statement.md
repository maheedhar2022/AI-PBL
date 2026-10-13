# Problem Statement

## Background

Road traffic accidents are a leading cause of fatalities and injuries worldwide.
Motorcycle riders are among the most vulnerable road users. Two specific violations
contribute significantly to rider fatalities:

1. **Riding without a helmet** — reduces protection in a crash by a factor of 3–4x.
2. **Triple riding** — carrying more than two persons on a motorcycle increases
   instability, reduces stopping distance, and overloads the vehicle.

Manual monitoring of traffic camera footage for these violations is labour-intensive,
inconsistent, and difficult to scale to 24/7 operation.

## Objective

Develop an AI-powered system that:

- Processes live traffic camera footage automatically.
- Detects motorcycles, riders, and helmet-related violations.
- Detects motorcycles carrying more than two riders.
- Records incident evidence and timestamps for human review.
- Enables research into AI model performance for this domain.

## Scope

- **In scope**: Motorcycles, helmets, rider count, live and recorded video.
- **Out of scope**: Car violations, pedestrian behaviour, audio analysis, GPS tracking.

## Constraints

- Must run without a paid cloud AI service.
- Must support CPU-only inference as a fallback.
- All detections are potential violations — human review is required.
- Standard COCO-trained models do not include helmet classes.
  A custom-trained model is required for production helmet detection.

## Success Criteria

1. The system correctly detects motorcycles and persons in standard test videos.
2. Triple-riding events (≥3 riders) are flagged with >80% precision on labeled data
   (after model evaluation with a real dataset).
3. Incident evidence is saved and reviewable in the dashboard.
4. The evaluation pipeline produces reproducible metrics from labeled data.

