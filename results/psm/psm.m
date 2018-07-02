
%----------------------------------------
%preallocates the hyperparameters
%----------------------------------------
N=10^4; %number of samples
par=23; %parameters
nproc=28;
%----------------------------------------
%Saves true parameters
%----------------------------------------
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
% Creates priors according to Anamika
%----------------------------------------
vs_unpert  = [3529; 3705; 3882; 3911; 4705; 4794; 4823]'; % Unperturbed v_s
rho_unpert = [2500*ones(1,4) 2900*ones(1,3)];             % Dito rho

% Prescribe uncertainties in the material properties.

layers=7;
alpha = 0.05;   % Uncertainty in v_s (multiplicative)
aa = 1.64;       % Lower bound on v_p/v_s
bb = 1.78;       % Upper bound on v_p/v_s
rho_low = 0.9;  % Lower bound on multiplicative uncertainty factor in rho
rho_high = 1.1; % Upper dito
Rl=rho_unpert*rho_low;
Ru=rho_unpert*rho_high;
vs_L = vs_unpert.*(1-alpha);
vs_H = vs_unpert.*(1+alpha);
vp_L = vs_L*aa;
vp_H = vs_H*bb;
%defines  maximum b_int and minimum a_int values for the material properties
a_int=[Rl;vp_L;vs_L];a_int=a_int(:)';
b_int=[Ru;vp_H;vs_H];b_int=b_int(:)';

% adaptivity 0 no, 1 yes
adapt=1;

%----------------------------------------
%Obtains Data
%----------------------------------------

beta=[1,10,100,10&8];
Nt=length(beta);
nx=58;
nz=34;
dt=10;
nx_true=nx*4;
nz_true=nz*4;
obs=1700;
dt_true=10/4;
X=cell(Nt,1);
Ns=1;
%Checks if datafile exist
filename=('original_data.mat');
if exist(filename,'file')
    data=load(filename);
    d=data.d;
    %dm=data.dm;
else
    disp('Data file not found. Generating data...')
    %0 means no movie. 1 otherwise.
    d=specfem(p_true,nx_true,nz_true,dt_true,obs,nproc);
    save(filename,'d');
end
%Perturbs the data
[dn,dm]=size(d);
noise=1E-3;
dp=d;
d=d+noise*randn(dn,dm);
DT=(1/obs);
%---------------------------------------------------------------------
%
% solver stuff
%
%---------------------------------------------------------------------
a=[20000,20000,a_int];
b=[120000,75000,b_int];
aa=a(3:end);
bb=b(3:end);
prior=@(x) 1*(prod(unifpdf(x,a,b))>0);
lik=@(f) -0.5*DT*sum((f'-d).^2)/(noise^2);
post=@(x,f) lik(f)+log(prior(x));

%%---------------------------------------------------------------------
%
% PSM parameters
%
%%---------------------------------------------------------------------
M =1; %number of samples
y=zeros(M,par);
x=zeros(M,par);
Rx=zeros(M,length(d));
Ry=zeros(M,length(d));

%samples first value
for i=1:M
x(i,:)=[50000,50000,aa+(bb-aa).*rand(1,par)];
Rx(i,:)=specfem(x(i,:),nx,nz,dt,obs,nproc) ;
end


px=zeros(M,1);
py=zeros(M,1);

ratio=zeros(1,M);
for i=1:M
    px(i,1)=post(X{i}(1,:),Rx(i,:));
end
%computes the mean PX
PX=mean(px);
disp('Entered MH loop')
acpt=zeros(N,1);
%creates matrix of initial covariances
X=zeros(N,2);
X(1,:)=mean(x(i,:));
Y=zeros(N,2);
S=eye(2); S(1,1)=1000; S(2,2)=1000;

for j=1:N
    
 
   %samples both x and Y 
   
for k=1:M
    P=aa+(bb-aa).*rand(1,par);
	y(k,:)=[X(i,:)+(S*randn(2,1))',P];
    x(k,:)=[X(i,:),P];
end
    
%Computes Rx , PX,  Ry and Py
    for i=1:M
        Ry(i,:)=specfem(y(i,:),nx,nz,dt,obs,nproc) ;
        py(i,:)=post(y(i,:),Ry(i,:));
        Rx(i,:)=specfem(x(i,:),nx,nz,dt,obs,nproc) ;
        px(i,:)=post(x(i,:),Rx(i,:));
    end
    Y=mean(y(:,1:2));
    PX=mean(px);
    PY=mean(py);
    ratio=PY-PX;
    %accepts-rejects
    
        if log(rand)<ratio
            X(j+1,:)=Y;
            acpt=acpt+1;
        else
            X(j+1,:)=X(j,:);
        end
    disp([' ratio ',num2str(exp(ratio))])
    filename = 'Psm_M1.mat';
   	save(filename)     
    disp(['acceptance at iteration ',num2str(j),' ',num2str(sum(acpt)/j),' current ',num2str(X(j+1,:))]);        
        
end
    



