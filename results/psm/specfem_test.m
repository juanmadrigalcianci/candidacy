function specfem_test
T=zeros(28,1);
for nproc=1:28
par=23; %parameters
X=zeros(1,par);
%xs = 51500 + Width of PML; zs = 57000 + Width of PML
X(1)=54000;
X(2)=59500;
%MATERIAL 1
X(3)=2571.3237764085998;
X(4)=6128.6721584572;
X(5)=3459.6956863028004;
%MATERIAL 2
X(6)=2426.41503611;
X(7)=6355.122874946;
X(8)=3799.8148401117;
%MATERIAL 3
X(9)=2520.1160128747;
X(10)=6799.444019500699;
X(11)=3823.0108244928997;
%MATERIAL 4
X(12)=2599.5359253555;
X(13)=6858.268976329799;
X(14)=3985.4579096294;
%MATERIAL 5
X(15)=2972.6688776596;
X(16)=7906.5743439161;
X(17)=4673.3570083854;
%MATERIAL 6
X(18)=3076.0623054293;
X(19)=8424.094384693199;
X(20)=4928.330648029099;
%MATERIAL 7
X(21)=3060.2828745172997;
X(22)=8434.0837880401;
X(23)=4999.1367830875;
p_true=X;

%----------------------------------------
%Obtains Data
%----------------------------------------
nx=58;
nz=34;
nx_true=nx*4;
nz_true=nz*4;
obs=1700;
dt_true=10/4;
tic;
specfem(p_true,nx_true,nz_true,dt_true,obs,nproc);
T(nproc)=toc;
filename=('time_per_processor.mat');
save(filename);
end




