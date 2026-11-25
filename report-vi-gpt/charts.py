import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
p=Path(__file__).parent/'figures'; p.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
def save(fig,name):
 fig.tight_layout(); fig.savefig(p/(name+'.svg'),bbox_inches='tight'); fig.savefig(p/(name+'.pdf'),bbox_inches='tight'); plt.close(fig)
def bars(name,labels,vals,xlabel,title,log=False):
 fig,ax=plt.subplots(figsize=(7,3.2)); y=np.arange(len(labels)); ax.barh(y,vals,color='#37678a'); ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlabel(xlabel); ax.set_title(title); ax.spines[['top','right']].set_visible(False); ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
 if log: ax.set_xscale('log'); ax.set_xlim(1000,1500000)
 else: ax.set_xlim(0,max(vals)*1.18)
 for i,v in enumerate(vals): ax.text(v*1.02,i,f'{v:,}'.replace(',',' '),va='center',fontsize=9)
 save(fig,name)
bars('01_nodes',['Minimax','Alpha–Beta tự nhiên','Alpha–Beta sắp xếp','Alpha–Beta + TT'],[549946,20866,7865,2172],'Số nút (thang log)','Tic-Tac-Toe 3×3, độ sâu 9',True)
fig,ax=plt.subplots(figsize=(4,4)); data={2:42,3:47,5:55,7:61,8:795}; ax.set_xlim(0,3); ax.set_ylim(3,0); ax.set_aspect('equal'); ax.axis('off')
for i in range(9):
 x,y=i%3,i//3; ax.add_patch(plt.Rectangle((x+.025,y+.025),.95,.95,color=('#287a62' if i==8 else '#59738a' if i in data else '#e7ebef'))); ax.text(x+.5,y+.35,f'Ô {i+1}',ha='center',color='white' if i in data else '#333');
 if i in data: ax.text(x+.5,y+.67,str(data[i]),ha='center',color='white',weight='bold',fontsize=16)
save(fig,'02_root')
bars('03_rollout_win',['Random','Heuristic','Value-Cutoff'],[16,34,74],'Tỷ lệ thắng (%)','Caro 9×9 — 1000 vòng lặp')
bars('04_rollout_time',['Random','Heuristic','Value-Cutoff'],[225,6645,1077],'Thời gian (ms/nước)','Caro 9×9 — 1000 vòng lặp')
bars('05_uct',['Greedy','Linear','No-Ln','No-Sqrt','Standard UCT','UCB1-Tuned','PUCT'],[44,50,50,16,28,42,98],'Tỷ lệ thắng (%)','Caro — 500 vòng lặp')
bars('06_xiangqi',['AB vs Random','MCTS-Random vs Random','MCTS-Heuristic vs Random','MCTS-ValueCut vs Random','PUCT-ValueCut vs Greedy','AB vs Greedy'],[88,66,88,96,90,50],'Tỷ lệ thắng (%)','Cờ tướng — 50 ván')
bars('07_delivery',['Nearest Neighbor','EDF','MCTS-Heuristic'],[2548.7,3592.1,1436.8],'Tổng chi phí','Định tuyến giao hàng — 16 đơn')
