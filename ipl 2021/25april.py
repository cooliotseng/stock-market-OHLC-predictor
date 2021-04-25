import random
raj = ['sanju', 'jos', 'riyan', 'david',   'morris', 'tewatia',
       'shivam', 'jaydev/shreyas', 'mustafizur/tye', 'jaiswal', 'chetan']
kkr = ['Nitish Rana.', 'Shubman Gill.', 'Rahul Tripathi.', 'Eoin Morgan', 'Andre Russell.', 'Dinesh Karthik ',
       'Shakib Al Hasan', 'Pat Cummins.', 'narine', 'Prasidh Krishna', ' Varun Chakravarthy']

mi = ['Rohit Sharma', 'Quinton de Kock', 'Suryakumar Yadav', 'Ishan Kishan', 'Kieron Pollard', 'Hardik Pandya',
      'Krunal Pandya', 'Jayant Yadav/Anukul Roy', 'Rahul Chahar', 'Trent Boult ', 'Jasprit Bumrah']
csk = ['Faf du Plessis', 'Ruturaj Gaikwad.', 'Ambati Rayudu', 'Suresh Raina', 'Sam Curran.',
       'M.S. Dhoni', 'Ravindra Jadeja', 'Moeen Ali', 'Shardul Thakur', 'Deepak Chahar', 'Bravo']
pbks = ['Nicholas Pooran', ' Chris Gayle', ' Mayank Agarwal', 'KL Rahul', 'Moises Henriques', ' Deepak Hooda', ' Mandeep Singh'
        ]
dc = ['Rishabh Pant', 'Ravichandran Ashwin', 'Avesh Khan/billings', 'Shikhar Dhawan', 'smith'
      'Shimron Hetmyer', 'Prithvi Shaw', 'Ajinkya Rahane', 'woakes','lalit', 'Amit Mishra/rabada']
rcb = ['Virat Kohli', 'AB de Villiers', 'Yuzvendra Chahal', 'Devdutt Padikkal', 'Ahmed/patel','kyle jamieson/rajat patidar',
        'Washington Sundar/christian', ' Mohammed Siraj', 'Kane Richardson', 'Glenn Maxwell', 'Adam Zampa/saini']
hyd = ['David Warner', 'Abhishek Sharma','pandey', 'samad', 'Jonny Bairstow', 'Manish Pandey',
       'singh/shankar', 'Goswami/holder', 'willamson/nabi', 'Mohammad Nabi/jadhav', 'roy/rashid', ' Wriddhiman Saha/kumar']
   
a  = dc+hyd

team3 = random.sample(a, 11)
team2 = []
(random.shuffle(a))



for i in a:
    if i not in team3:
        team2.append(i)
    else:
        pass


team1 = random.sample(team3, 5)
team1 += random.sample(team2, 6)
print('team3', team3)
print('team2', team2)

print('team1', team1)
print('team3', len(team3))
print('team2', len(team2))

print('team1', len(team1))
