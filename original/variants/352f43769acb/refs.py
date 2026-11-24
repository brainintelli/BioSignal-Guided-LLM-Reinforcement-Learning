from data import O
items=[
('kagan2022','Brett J. Kagan and Andy C. Kitchen and Nhi T. Tran and Forough Habibollahi and Moein Khajehnejad and Bradyn J. Parker and Anjali Bhat and Ben Rollo and Adeel Razi and Karl J. Friston','In vitro neurons learn and exhibit sentience when embodied in a simulated game-world','Neuron','2022','10.1016/j.neuron.2022.09.001',''),
('smirnova2023','Lena Smirnova and Brian S. Caffo and David H. Gracias and Qi Huang and Itzy E. Morales Pantoja and Bohao Tang and Donald J. Zack and Cynthia A. Berlinicke and J. Lomax Boyd and Timothy D. Harris and Erik C. Johnson and Brett J. Kagan and Jeffrey Kahn and Alysson R. Muotri and Barton L. Paulhamus and Jens C. Schwamborn and Jesse Plotkin and Alexander S. Szalay and Joshua T. Vogelstein and Paul F. Worley and Thomas Hartung','Organoid intelligence (OI): the new frontier in biocomputing and intelligence-in-a-dish','Frontiers in Science','2023','10.3389/fsci.2023.1017235',''),
('wang2023','Guanzhi Wang and Yuqi Xie and Yunfan Jiang and Ajay Mandlekar and Chaowei Xiao and Yuke Zhu and Linxi Fan and Anima Anandkumar','Voyager: An Open-Ended Embodied Agent with Large Language Models','arXiv preprint','2023','','2305.16291'),
('fortunato2018','Meire Fortunato and Mohammad Gheshlaghi Azar and Bilal Piot and Jacob Menick and Ian Osband and Alex Graves and Vlad Mnih and Remi Munos and Demis Hassabis and Olivier Pietquin and Charles Blundell and Shane Legg','Noisy Networks for Exploration','International Conference on Learning Representations','2018','','1706.10295'),
('schulman2017','John Schulman and Filip Wolski and Prafulla Dhariwal and Alec Radford and Oleg Klimov','Proximal Policy Optimization Algorithms','arXiv preprint','2017','','1707.06347')]
bib=[]
for key,a,t,j,y,doi,arx in items:
 typ='inproceedings' if key=='fortunato2018' else 'article'
 kv={'author':a,'title':'{'+t+'}','booktitle' if typ=='inproceedings' else 'journal':j,'year':y}
 if doi:kv.update(doi=doi,url='https://doi.org/'+doi)
 else:kv.update(eprint=arx,archivePrefix='arXiv',url='https://arxiv.org/abs/'+arx)
 if key=='kagan2022':kv.update(volume='110',number='23',pages='3952--3969.e8')
 if key=='smirnova2023':kv.update(volume='1',pages='1017235')
 bib.append('@'+typ+'{'+key+',\n'+',\n'.join('  '+k+' = {'+v+'}' for k,v in kv.items())+'\n}')
(O/'refs.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf-8')
