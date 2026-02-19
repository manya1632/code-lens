{"code":"if user.is_admin == True:\n    allow()","language":"python","bugs":["style"],"severity":0.2,"complexity_before":"O(1)","complexity_after":"O(1)"}
{"code":"data = open('file.txt').read()","language":"python","bugs":["security"],"severity":0.6,"complexity_before":"O(n)","complexity_after":"O(n)"}
{"code":"for i in range(n):\n    for j in range(n):\n        for k in range(n):\n            print(i,j,k)","language":"python","bugs":["performance"],"severity":0.9,"complexity_before":"O(n³)","complexity_after":"O(n²)"}
{"code":"try:\n    risky()\nexcept:\n    pass","language":"python","bugs":["logic"],"severity":0.7,"complexity_before":"O(1)","complexity_after":"O(1)"}
