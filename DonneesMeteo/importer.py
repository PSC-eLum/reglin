import csv 
import datetime
    
def input() :
    csvDoc = input("Adresse complète du csv : ")
    #return fichier(csvDoc)
    
def fichier(fName) :
    #creation du nouveau fichier
    tab = []
    
    #parcours du nouveau truc
    file = open(fName, "rb")
    
    for row in csv.DictReader(file)
        # Avant la lecture d'un enregistrement le nom des champs n'est
        # pas disponible.
        print "Titres (avant next()):", reader.fieldnames 
    
        # La lecture du premier enregistrement...
        row = reader.next()
    
        # ...initialise la liste des titre de colonnes:
        print "Titres (après next()):", reader.fieldnames
        print "Data row 1:", row
    
    file.close()
#             #conversion du temps
#             n = len(total[0])
#             time.insert(n,datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"))
#             for i in range(1, len(total)):
#                 total[i].insert(n,row[i]);
#             
# for row in csv.reader(f):
#     n = len(total[0])
#     time.insert(n,datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"))
#     for i in range(1, len(total)):
#         total[i].insert(n,row[i]);