
import json

a = 0


with open('iBlurry/collections/cifar10/cifar10_split5_n50_m10_rand1_task0.json', 'r') as file:
    data = json.load(file)
    classes =[]
    count = 0
    for d in data:
        classes.append(d['klass'])
        if d['klass'] == 'deer':
            count+=1

classes = set(classes)
a += count
print(count)
print(classes)

with open('iBlurry/collections/cifar10/cifar10_split5_n50_m10_rand1_task1.json', 'r') as file:
    data = json.load(file)
    classes =[]
    count = 0
    for d in data:
        classes.append(d['klass'])
        if d['klass'] == 'cat':
            count+=1

classes = set(classes)
a += count
print(count)
print(classes)

with open('iBlurry/collections/cifar10/cifar10_split5_n50_m10_rand1_task2.json', 'r') as file:
    data = json.load(file)
    classes =[]
    count = 0
    for d in data:
        classes.append(d['klass'])
        if d['klass'] == 'dog':
            count+=1

classes = set(classes)
a += count
print(count)
print(classes)


with open('iBlurry/collections/cifar10/cifar10_split5_n50_m10_rand1_task3.json', 'r') as file:
    data = json.load(file)
    classes =[]
    count = 0
    for d in data:
        classes.append(d['klass'])
        if d['klass'] == 'truck':
            count+=1

classes = set(classes)
a += count
print(count)
print(classes)

with open('iBlurry/collections/cifar10/cifar10_split5_n50_m10_rand1_task4.json', 'r') as file:
    data = json.load(file)
    classes =[]
    count=0
    for d in data:
        classes.append(d['klass'])
        if d['klass'] == 'bird':
            count+=1

classes = set(classes)
a += count
print(count)
print(classes)

print(a)