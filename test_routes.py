from app import app

client = app.test_client()
routes = ['/', '/data', '/eda', '/model', '/student/S001', '/predict', '/concept']

for r in routes:
    res = client.get(r)
    print(f'{r:20s} -> {res.status_code}')
    assert res.status_code == 200, f'Failed on {r}'

post_res = client.post('/predict', data={
    'study_hours_week': 4,
    'attendance_pct': 60,
    'prev_grade': 8,
    'absences': 12,
    'past_failures': 2,
    'assignment_pct': 50,
    'sleep_hours': 5,
    'parental_support': 2,
    'internet_access': 1
})
print(f"{'POST /predict':20s} -> {post_res.status_code}")
assert post_res.status_code == 200

for d in ['/docs/Concept_Note_final_ibm_project.docx', '/docs/Final_Deliverable_PPT_ibm.pptx']:
    res = client.get(d)
    print(f'{d:45s} -> {res.status_code}')
    assert res.status_code == 200

print("\n>>> ALL FRONTEND AND BACKEND ROUTES WORK PERFECTLY! <<<")
