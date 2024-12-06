dic = {'hi': [1]}

if 'hi' not in dic.keys():
    dic['hi'] = [1]
else:
    dic['hi'].append(2)

print(dic)