import pandas as pd
import numpy as np

total_visits = 1005
device_distribution = {'desktop': 0.5114, 'mobile': 0.4886}
average_session_duration = 300
average_pages_per_session = 3
days_in_month = 31
hours_in_day = 24

np.random.seed(42)

visits_per_hour = np.random.poisson(lam=total_visits / (days_in_month * hours_in_day), size=days_in_month * hours_in_day)
visits_per_hour = visits_per_hour.clip(0, 10)

pages_per_session = np.random.normal(average_pages_per_session, 0.5, total_visits).astype(int).clip(1, 10)
session_duration = np.random.normal(average_session_duration, 50, total_visits).astype(int).clip(60, 1000)

data = {
    'date': [],
    'hour': [],
    'device': [],
    'session_duration': [],
    'pages_per_session': [],
}

visit_index = 0
for day in range(1, days_in_month + 1):
    for hour in range(hours_in_day):
        for _ in range(visits_per_hour[(day - 1) * hours_in_day + hour]):
            data['date'].append(f'2024-10-{day:02d}')
            data['hour'].append(f'{hour:02d}:00')
            data['device'].append(np.random.choice(['desktop', 'mobile'], p=[device_distribution['desktop'], device_distribution['mobile']]))
            data['session_duration'].append(session_duration[visit_index])
            data['pages_per_session'].append(pages_per_session[visit_index])
            visit_index += 1

df = pd.DataFrame(data)
df.to_excel('website_visits.xlsx', index=False)