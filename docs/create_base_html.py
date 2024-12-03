import json
import os

cur_directory = os.path.dirname(__file__)

with open(os.path.join(cur_directory, '_static', 'switcher.json'), 'r') as file:
    data = json.load(file)
preferred_entry = next((item for item in data if item.get('preferred')), None)
if preferred_entry:
    redirect_url = preferred_entry['url']
else:
    assert False and 'no preferred'

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="0;url={redirect_url}">
    <title>Redirecting...</title>
</head>
<body>
    <p><a href="{redirect_url}">redirect to stable</a></p>
</body>
</html>
"""
with open(os.path.join(cur_directory, os.path.pardir, 'pages', 'index.html'), 'w') as file:
    file.write(html_content)
