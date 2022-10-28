def csv_to_txt(path):
    ''' Function that convert a csv in a txt '''

    lines = []
    newlines = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            newlines.append(line.replace(',',' '))
            
    with open('newfile.txt', 'w') as f:
        f.writelines(newlines)
                

csv_to_txt('WMATA.csv')