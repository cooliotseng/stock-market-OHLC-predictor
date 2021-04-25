import random
raj = ['sanju', 'jos', 'riyan', 'david',   'morris', 'tewatia',
       'shivam', 'jaydev/shreyas', 'mustafizur/tye', 'jaiswal', 'chetan']
kkr = ['Nitish Rana.', 'Shubman Gill.', 'Rahul Tripathi.', 'Eoin Morgan', 'Andre Russell.', 'Dinesh Karthik ',
       'Shakib Al Hasan', 'Pat Cummins.', 'narine', 'Prasidh Krishna', ' Varun Chakravarthy']

mi = ['Rohit Sharma', 'Quinton de Kock', 'Suryakumar Yadav', 'Ishan Kishan', 'Kieron Pollard', 'Hardik Pandya',
      'Krunal Pandya', 'Jayant Yadav/Anukul Roy', 'Rahul Chahar', 'Trent Boult ', 'Jasprit Bumrah']
csk = ['Faf du Plessis', 'Ruturaj Gaikwad.', 'Ambati Rayudu', 'Suresh Raina', 'Sam Curran.',
       'M.S. Dhoni', 'Ravindra Jadeja', 'Moeen Ali', 'Shardul Thakur', 'Deepak Chahar', 'Imran Tahir']
pbks = ['Nicholas Pooran', ' Chris Gayle', ' Mayank Agarwal', 'KL Rahul', 'Moises Henriques', ' Deepak Hooda', ' Mandeep Singh'
        ]
dc = ['Rishabh Pant', 'Ravichandran Ashwin', 'Avesh Khan', 'Shikhar Dhawan', 'Praveen Dubey',
      'Shimron Hetmyer', 'Prithvi Shaw', 'Ajinkya Rahane', 'Shreyas Iyer', ' Axar Patel', 'Amit Mishra']
rcb = ['Virat Kohli', 'AB de Villiers', 'Yuzvendra Chahal', 'Devdutt Padikkal', 'Devdutt Padikkal',
       'Navdeep Saini', 'Washington Sundar', ' Mohammed Siraj', 'Kane Richardson', 'Glenn Maxwell', 'Adam Zampa']
hyd = ['David Warner', 'Abhishek Sharma', 'Bhuvneshwar Kumar', 'Jonny Bairstow', 'Manish Pandey',
       'Kane Williamson', 'T Natarajan', 'Rashid Khan', 'Mohammad Nabi', 'Sandeep Sharma', ' Wriddhiman Saha']
   
a  = raj+kkr

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
