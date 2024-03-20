total_string = ""

for num in range(365):
    file = open(f'../major_data/major_data_{num}.txt', 'r')
    text = file.read()

    lines = text.split('\n')
    major_name = lines[0]

    arr = major_name.split('/')

    major_name = arr[-1]

    total_string+=f"{major_name}\n"

    new_file = open(f'../renamed_data/{major_name}.txt', 'a')

    new_file.write(text)

    file.close()
    new_file.close()

file2 = open('../major_names.txt', 'a')

file2.write(total_string)

file2.close()
