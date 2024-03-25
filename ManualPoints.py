import numpy as np

pointsTestImage0 = np.array([  [46.940006	,135.86835],
                                [52.815453	,135.80544],
                                [53.953236	,129.6532 ],
                                [72.303246	,110.77625],
                                [88.84565	,103.18099],
                                [89.341072	,110.35077],
                                [127.06176	,80.096748],
                                [134.8576	,147.94672],
                                [141.55338	,147.78088],
                                [141.75244	,165.61325],
                                [142.94322	,155.72792],
                                [149.03343	,142.01109],
                                [149.5491	,140.14581],
                                [150.76295	,179.22363],
                                [155.17903	,133.14845],
                                [155.24362	,176.43748],
                                [155.99936	,167.87552],
                                [162.03705	,133.48051],
                                [167.69789	,133.29654],
                                [171.53683	,136.7659 ],
                                [174.23016	,175.70522],
                                [174.8914	,209.79115],
                                [178.85426	,208.90184],
                                [180.11299	,168.13602],
                                [183.1346	,190.05872],
                                [185.9791	,178.49809],
                                [187.90248	,176.75476],
                                [189.02441	,169.84842],
                                [196.71703	,198.05322],
                                [201.31258	,203.87956],
                                [202.86069	,194.32069],
                                [214.24677	,74.764183],
                                [217.31192	,129.94951],
                                [227.59531	,122.94302],
                                [233.33195	,153.0027 ],
                                [234.0945	,69.609566],
                                [233.87364	,147.95193],
                                [234.84079	,161.24173],
                                [244.61429	,186.60031],
                                [247.61066	,101.05563],
                                [248.06342	,107.03111],
                                [251.14786	,219.19949],
                                [256.66708	,87.303543],
                                [261.23126	,85.608749],
                                [273.93713	,102.25535],
                                [281.97995	,149.38887],
                                [282.30057	,153.04256],
                                [283.80438	,102.46359],
                                [287.94547	,150.07999],
                                [288.35059	,214.51062],
                                [292.44278	,223.85262],
                                [293.85248	,228.04729],
                                [301.38794	,109.93826],
                                [302.04947	,92.309166],
                                [303.69067	,122.9575 ],
                                [306.32367	,116.06443],
                                [307.50192	,146.08633],
                                [306.95013	,196.46208],
                                [308.95117	,162.4024 ],
                                [309.56149	,210.05087],
                                [310.32889	,177.72899],
                                [310.78827	,123.70577],
                                [311.57343	,115.05538],
                                [313.68304	,198.73311],
                                [314.09561	,207.49748],
                                [315.05328	,150.1246 ],
                                [316.50098	,187.838  ],
                                [315.65588	,225.02301],
                                [316.54904	,123.01808],
                                [317.90125	,126.78337],
                                [318.17279	,145.27303],
                                [320.18134	,199.17844],
                                [320.74115	,185.4437 ],
                                [321.51144	,102.63887],
                                [322.10361	,114.96196],
                                [324.83871	,140.5787 ],
                                [326.52689	,99.605171],
                                [327.41263	,192.65311],
                                [327.59283	,126.02861],
                                [333.26212	,144.97351],
                                [337.3129	,148.70537],
                                [337.10059	,183.62564],
                                [337.93808	,187.94429],
                                [341.79065	,105.13363],
                                [343.21347	,158.8232 ],
                                [344.10428	,166.94588],
                                [345.9295	,156.12738],
                                [347.02933	,205.55424],
                                [353.19519	,113.14326],
                                [359.56427	,203.96471],
                                [360.70618	,111.12277],
                                [368.95782	,115.52012],
                                [373.98434	,202.86617],
                                [374.09579	,248.92531],
                                [375.07477	,148.04286],
                                [384.51105	,147.60921],
                                [384.73193	,201.74881],
                                [386.50696	,157.82547],
                                [392.34372	,220.5674 ],
                                [392.54034	,226.27139],
                                [393.2002	,227.94949],
                                [395.38828	,174.98228],
                                [396.24036	,154.7916 ],
                                [396.30051	,161.59741],
                                [398.23798	,182.53099],
                                [397.88846	,214.81769],
                                [399.39508	,143.89722],
                                [398.87958	,208.16432],
                                [400.7543	,151.05598],
                                [401.38425	,154.78708],
                                [403.38272	,136.70291],
                                [402.94373	,159.26878],
                                [403.00421	,230.08342],
                                [404.40903	,189.63457],
                                [408.09012	,212.55434],
                                [409.6015	,136.64391],
                                [411.44464	,225.61577],
                                [413.81421	,141.86812],
                                [414.86765	,159.8253 ],
                                [414.96774	,198.34264],
                                [415.45239	,222.5285 ],
                                [417.64102	,218.57722],
                                [418.15164	,150.25708],
                                [418.7988	,208.91179],
                                [421.03024	,163.42752],
                                [422.50339	,143.33719],
                                [422.0596	,217.70474],
                                [427.22491	,184.68938],
                                [427.52829	,158.9575 ],
                                [429.34616	,216.90967],
                                [430.62793	,183.35292],
                                [435.95032	,162.78043],
                                [437.49765	,194.97107],
                                [438.00195	,180.40961],
                                [439.42661	,204.25366],
                                [442.88303	,201.20436],
                                [444.27603	,163.22816],
                                [443.73135	,180.91037],
                                [444.01395	,216.39394],
                                [445.21945	,171.51431],
                                [446.78305	,143.1846 ],
                                [448.3299	,173.98801],
                                [449.58203	,153.04961],
                                [450.13977	,180.81754],
                                [449.66431	,207.76564],
                                [454.12363	,156.23859],
                                [457.34378	,208.83266],
                                [460.3497	,139.96417],
                                [461.66119	,90.094025],
                                [462.8129	,171.2235 ],
                                [466.76273	,52.838715],
                                [468.37277	,87.485741],
                                [468.51511	,91.464325],
                                [467.72293	,211.30042],
                                [468.45395	,195.24335],
                                [469.74997	,148.14746],
                                [469.59436	,152.52332],
                                [471.14713	,85.005814],
                                [475.76825	,192.28816],
                                [476.80698	,182.63925],
                                [478.3241	,163.03664],
                                [478.18445	,178.51833],
                                [478.65823	,152.49699],
                                [480.71658	,136.58182],
                                [481.06918	,168.71242],
                                [485.13541	,137.99075],
                                [484.82364	,153.84642],
                                [487.02338	,170.841  ],
                                [488.18234	,180.82417],
                                [488.58218	,194.67369],
                                [489.88327	,154.03532],
                                [492.00104	,168.57727],
                                [494.38498	,123.24506],
                                [493.79926	,145.91779],
                                [493.87271	,180.77913],
                                [494.59766	,138.89133],
                                [496.06769	,154.75473],
                                [496.11536	,190.74585],
                                [495.65918	,194.11769],
                                [497.41681	,149.76328],
                                [499.33688	,160.36021],
                                [499.09366	,180.70624],
                                [498.45618	,185.77072],
                                [501.02933	,175.21722],
                                [502.89502	,188.19472],
                                [510.01932	,189.85527],
                                [517.24225	,178.33032],
                                [521.02246	,169.71089],
                                [522.98712	,177.87379],
                                [529.60773	,185.41246],
                                [531.78766	,203.28645],
                                [535.15533	,183.2428 ],
                                [535.93964	,194.0918 ],
                                [542.8111	,163.57074],
                                [542.8623	,183.03621],
                                [543.44055	,212.21712],
                                [543.77203	,204.64575],
                                [544.82422	,218.23628],
                                [547.93799	,195.68059],
                                [556.47797	,165.64275],
                                [565.76544	,131.06467],
                                [567.53717	,141.02545],
                                [579.86835	,175.37027],
                                [583.72711	,97.277084],
                                [584.7384	,94.588814],
                                [585.12292	,105.41789],
                                [586.47064	,109.11592],
                                [588.71729	,80.359184],
                                [589.28693	,103.49337],
                                [590.95782	,112.24297],
                                [599.13751	,147.94308],
                                [599.51941	,165.90535],
                                [610.50293	,147.44928],
                                [610.04663	,154.77213],
                                [610.49774	,165.18446],
                                [614.78839	,51.813972],
                                [630.1734	,56.77652 ],
                                [630.25464	,62.082432],
                                [631.58295	,220.278  ],
                                [641.02405	,125.15632],
                                [641.23804	,133.07988],
                                [642.56879	,221.56407],
                                [644.07013	,40.322842],
                                [643.58167	,155.15802],
                                [646.1012	,26.321098],
                                [645.88196	,126.17418],
                                [646.7774	,140.56264],
                                [647.33069	,148.0318 ],
                                [647.5257	,229.78395],
                                [648.39673	,33.880554],
                                [661.25269	,55.536022],
                                [668.36938	,58.681061],
                                [669.61353	,34.450699],
                                [671.48749	,247.92899],
                                [672.58362	,42.377979],
                                [672.10046	,101.92783],
                                [711.78302	,85.992905],
                                [725.64838	,67.48822 ],
                                [748.88892	,273.09732],
                                [766.88605	,80.322205],
                                [770.28485	,147.75542],
                                [771.3653	,103.08261],
                                [777.01154	,28.93294 ],
                                [778.16602	,106.14343],
                                [777.71893	,119.11616],
                                [781.40643	,64.895279],
                                [780.3595	,83.261147],
                                [782.04993	,69.191719],
                                [783.46753	,28.315559],
                                [784.29968	,95.281448],
                                [785.22681	,66.840065],
                                [788.35229	,92.070961],
                                [790.38971	,82.212914],
                                [795.77942	,92.610802],
                                [796.34088	,94.313766],
                                [797.15833	,53.814163],
                                [799.90918	,86.432388],
                                [800.94189	,62.753265],
                                [800.36346	,81.680794],
                                [803.87939	,51.234497],
                                [805.20135	,151.56708],
                                [807.43982	,62.53241 ],
                                [807.30414	,72.472504],
                                [807.13757	,155.68341],
                                [808.46307	,128.46718],
                                [808.37964	,148.0005 ],
                                [811.22479	,132.42639],
                                [810.86005	,155.80298],
                                [811.64795	,148.89743],
                                [813.49512	,47.790565],
                                [814.29517	,134.15659],
                                [815.09528	,41.764484],
                                [815.16479	,81.735466],
                                [814.78326	,150.11478],
                                [819.22473	,36.363556],
                                [819.20697	,69.243813],
                                [820.19672	,93.458595],
                                [822.01947	,48.652908],
                                [822.21082	,97.839439],
                                [832.83264	,74.243561],
                                [834.37006	,154.89915],
                                [845.81519	,43.860821],
                                [850.2561	,42.459011],
                                [850.33398	,75.187592],
                                [851.43298	,88.706123],
                                [854.15161	,45.706181],
                                [854.79169	,88.807068],
                                [859.41974	,54.069374],
                                [866.00964	,54.988941],
                                [869.10101	,144.85126],
                                [868.95685	,149.25781],
                                [890.48584	,150.49756],
                                [893.93707	,146.63943],
                                [894.59479	,152.93498],
                                [902.46228	,293.46112],
                                [902.56781	,130.27774],
                                [902.8819	,141.51765],
                                [903.20563	,290.06943],
                                [908.90558	,292.3103 ],
                                [910.15948	,126.86958],
                                [914.11499	,151.74527],
                                [914.02924	,288.08902],
                                [916.92883	,111.93777],
                                [920.16248	,151.8121 ],
                                [924.85748	,287.03455],
                                [929.58563	,131.65312],
                                [928.56812	,307.78268],
                                [934.36816	,287.03156],
                                [938.27368	,90.739761],
                                [938.22418	,125.26649],
                                [938.15094	,130.49818],
                                [938.09967	,140.38463],
                                [938.96106	,151.00703],
                                [939.03943	,287.15161],
                                [941.43536	,159.45039],
                                [941.94751	,115.0935 ],
                                [942.2605	,118.3064 ],
                                [941.91644	,161.61349],
                                [943.35681	,29.870869],
                                [943.15295	,292.74484],
                                [943.51086	,60.644501],
                                [944.34863	,90.280594],
                                [947.03082	,71.976387],
                                [946.94092	,161.93163],
                                [950.91669	,119.77293],
                                [951.59467	,159.41675],
                                [957.78278	,158.54652],
                                [962.92786	,238.13817],
                                [969.2251	,257.85764],
                                [972.55542	,256.4693 ]], dtype=np.float32)





pointsTestImage1 = np.array([  [26.601564 ,137.55043],
                                [32.233238 ,137.37335],
                                [35.160912 ,130.1936 ],
                                [59.759552 ,111.79312],
                                [76.770485 ,103.96267],
                                [77.056068 ,111.44505],
                                [110.26299 ,79.570351],
                                [123.92899 ,149.65875],
                                [130.59526 ,149.78528],
                                [131.00014 ,167.87485],
                                [132.29108 ,158.017  ],
                                [138.52893 ,143.82471],
                                [138.96538 ,142.17665],
                                [140.36958 ,182.02301],
                                [144.48853 ,134.5242 ],
                                [145.09265 ,178.83766],
                                [145.58745 ,170.45306],
                                [151.8213  ,134.87129],
                                [157.70418 ,134.95213],
                                [161.44867 ,138.46815],
                                [164.75333 ,178.21265],
                                [165.10622 ,213.43845],
                                [169.86903 ,212.13437],
                                [170.59978 ,170.53589],
                                [173.56633 ,193.12773],
                                [176.4077  ,181.08142],
                                [178.30273 ,179.20491],
                                [179.26115 ,172.61249],
                                [187.7001  ,201.1304 ],
                                [192.60869 ,206.92821],
                                [193.89891 ,197.24001],
                                [201.353   ,73.175484],
                                [204.94688 ,131.2034 ],
                                [215.93658 ,123.81672],
                                [221.35786 ,154.57742],
                                [222.23235 ,67.416145],
                                [221.9207  ,149.12897],
                                [223.44983 ,162.76526],
                                [233.35281 ,189.34271],
                                [235.81929 ,100.61331],
                                [235.97133 ,106.44121],
                                [244.6174  ,222.71439],
                                [244.06178 ,85.932816],
                                [250.21364 ,84.129303],
                                [264.49533 ,101.92589],
                                [276.91534 ,151.21545],
                                [277.24921 ,155.02484],
                                [274.80444 ,102.01103],
                                [283.48129 ,151.89728],
                                [283.76422 ,217.68556],
                                [286.72974 ,227.78384],
                                [287.83148 ,231.89919],
                                [297.32858 ,110.87296],
                                [293.28442 ,91.200752],
                                [299.87268 ,124.05281],
                                [302.28458 ,117.06126],
                                [303.68866 ,148.00899],
                                [302.74597 ,199.22453],
                                [305.41083 ,164.49135],
                                [305.86234 ,212.68321],
                                [306.76886 ,180.08698],
                                [306.96097 ,124.79164],
                                [308.01697 ,116.06577],
                                [310.22562 ,201.40002],
                                [310.61725 ,210.00581],
                                [311.59454 ,151.8871 ],
                                [314       ,190      ],
                                [309.92685 ,228.57364],
                                [313.24942 ,124.2657 ],
                                [314.91431 ,128.06174],
                                [314.97162 ,147.06575],
                                [317.01578 ,201.877  ],
                                [317.80508 ,187.6799 ],
                                [318.46082 ,103.56171],
                                [319.08807 ,116.06843],
                                [321.92804 ,141.31627],
                                [323.82602 ,100.28771],
                                [324.58786 ,194.96356],
                                [324.81537 ,127.328  ],
                                [330.55319 ,146.59236],
                                [334.43414 ,150.46169],
                                [334.72531 ,185.92639],
                                [335.47073 ,190.29161],
                                [339.61591 ,106.25083],
                                [340.85556 ,160.71786],
                                [341.90793 ,168.97589],
                                [343.6391  ,158.0416 ],
                                [342.62271 ,208.42952],
                                [349.82352 ,113.33388],
                                [355.24731 ,206.75243],
                                [357.59354 ,111.79011],
                                [365.82419 ,116.59058],
                                [370.2178  ,205.58626],
                                [367.41022 ,253.79346],
                                [373.81186 ,149.68665],
                                [382.55814 ,148.50845],
                                [380.98099 ,204.44196],
                                [384.26163 ,159.55452],
                                [388.77118 ,223.81415],
                                [389.22464 ,229.27365],
                                [390.0379  ,231.55722],
                                [395.02246 ,177.09842],
                                [394.05063 ,156.6461 ],
                                [394.37271 ,162.80107],
                                [397.94485 ,184.87622],
                                [394.50543 ,217.91953],
                                [397.18811 ,145.78966],
                                [395.57709 ,211.14212],
                                [398.69403 ,153.1143 ],
                                [399.16141 ,156.25731],
                                [401.99487 ,137.30014],
                                [400.88623 ,160.92996],
                                [400.09366 ,233.4306 ],
                                [401.97086 ,192.00742],
                                [405.20532 ,215.51013],
                                [409.01611 ,137.68636],
                                [409.00366 ,228.76515],
                                [412.2247  ,143.20914],
                                [414.24039 ,161.55962],
                                [412.94467 ,200.70509],
                                [413.42737 ,225.53716],
                                [415.35104 ,221.55693],
                                [417.96353 ,151.98976],
                                [416.5603  ,211.63628],
                                [420.49811 ,164.79884],
                                [422.04141 ,144.74771],
                                [420.79675 ,220.24207],
                                [427.01138 ,186.62448],
                                [426.8671  ,160.87685],
                                [427.72574 ,219.93295],
                                [430.772   ,185.35269],
                                [436.63193 ,164.63635],
                                [436.57947 ,197.21901],
                                [438.68655 ,182.40425],
                                [438.83368 ,206.89684],
                                [441.39218 ,203.84013],
                                [445.18149 ,165.08385],
                                [444.59915 ,182.92589],
                                [442.8045  ,219.32091],
                                [446.05313 ,173.37881],
                                [447.12094 ,144.7851 ],
                                [449.45334 ,176.12753],
                                [450.04007 ,154.68858],
                                [450.75235 ,182.43985],
                                [448.69821 ,210.23273],
                                [454.59918 ,157.77466],
                                [457.16171 ,211.09439],
                                [461.3949  ,141.47334],
                                [460.8013  ,90.158562],
                                [464.17618 ,173.18242],
                                [466.15735 ,51.542393],
                                [468       ,88       ],
                                [467.86761 ,91.712173],
                                [467.95251 ,213.77179],
                                [468.42877 ,197.36064],
                                [470.52905 ,149.48846],
                                [470.79266 ,153.51004],
                                [470.78445 ,84.764839],
                                [475.92429 ,194.2968 ],
                                [478.13809 ,184.48297],
                                [479.58105 ,164.61322],
                                [479.09467 ,181.17154],
                                [479.73441 ,153.97168],
                                [481.92343 ,138.04964],
                                [482.06586 ,170.74271],
                                [486.60516 ,139.62894],
                                [486.30295 ,155.1102 ],
                                [488.69205 ,172.72789],
                                [490.02963 ,182.66318],
                                [489.80026 ,196.52707],
                                [491.43271 ,155.85765],
                                [493.89807 ,170.17529],
                                [496.28287 ,124.88531],
                                [495.37946 ,147.53635],
                                [495.62274 ,182.53104],
                                [496.17731 ,140.48065],
                                [496.69336 ,155.81808],
                                [497.45612 ,193.21985],
                                [496.6748  ,196.80846],
                                [499.04031 ,151.1033 ],
                                [500.16748 ,161.1282 ],
                                [500.80035 ,182.58714],
                                [499.9202  ,187.72035],
                                [502.83591 ,176.94463],
                                [504.28693 ,189.98193],
                                [511.59711 ,191.86203],
                                [519.21136 ,180.19052],
                                [522.896   ,171.26305],
                                [524.97034 ,179.57082],
                                [531.35211 ,187.36188],
                                [533.95221 ,205.29938],
                                [537.13434 ,184.95198],
                                [537.7063  ,195.7446 ],
                                [544.98572 ,165.086  ],
                                [544.78607 ,184.77066],
                                [545.30121 ,213.35524],
                                [545.7984  ,206.16841],
                                [546.75873 ,219.56587],
                                [549.83386 ,197.76192],
                                [558.77966 ,167.32875],
                                [567.56604 ,132.15222],
                                [569.47046 ,142.1416 ],
                                [581.96497 ,176.86131],
                                [585.45508 ,97.303841],
                                [586.52393 ,94.825333],
                                [586.8363  ,105.60321],
                                [588.27136 ,109.44254],
                                [590.68689 ,80.163391],
                                [590.49908 ,103.33133],
                                [592.8219  ,113.16393],
                                [601.46838 ,149.34358],
                                [601.67517 ,167.31053],
                                [612.62482 ,147.5887 ],
                                [611.97839 ,155.51596],
                                [613.10443 ,166.51462],
                                [616.9342  ,49.085228],
                                [633.21545 ,54.522038],
                                [633.04639 ,60.155891],
                                [634.3548  ,222.60941],
                                [643.62518 ,125.80604],
                                [643.64716 ,133.06833],
                                [646.24066 ,224.05482],
                                [647.8277  ,35.604855],
                                [646.26416 ,156.24568],
                                [649.64313 ,21.090033],
                                [648.70544 ,126.48462],
                                [649.82336 ,141.26759],
                                [650.21588 ,149.33284],
                                [651.06543 ,232.63858],
                                [652.19952 ,29.19997 ],
                                [665.84991 ,51.878941],
                                [673.91901 ,54.816357],
                                [674.35986 ,29.034664],
                                [676.79895 ,252.03485],
                                [677.43756 ,37.252323],
                                [676.21332 ,100.72765],
                                [718.85638 ,82.99231 ],
                                [733.55426 ,63.059811],
                                [760.76178 ,279.92914],
                                [776.40979 ,76.983109],
                                [777.71381 ,147.78166],
                                [781.1001 ,100.78419],
                                [786.87506 ,23.56567 ],
                                [788.00873 ,104.13754],
                                [787.69043 ,117.41815],
                                [792.92719 ,60.627934],
                                [791.03839 ,79.390839],
                                [792.29309 ,65.650131],
                                [793.86267 ,22.028955],
                                [794.3363 ,92.191422],
                                [795.36096 ,63.342827],
                                [798.30225 ,89.541801],
                                [800.85974 ,78.550262],
                                [807 ,89           ],
                                [807.2558 ,91.052826],
                                [807.88971 ,49.347824],
                                [811.13757 ,83.479515],
                                [812.50153 ,57.685543],
                                [811.8869 ,77.970772],
                                [815.47992 ,45.683655],
                                [814.3252 ,151.41354],
                                [819.1557 ,57.72467 ],
                                [819.60358 ,67.866402],
                                [816 ,156          ],
                                [816.96655 ,127.86942],
                                [817.04437 ,148.01152],
                                [819.93506 ,132.05759],
                                [819.50922 ,155.98361],
                                [820.2514 ,148.95795],
                                [826.492 ,42.625919],
                                [823.08197 ,133.56296],
                                [826.48438 ,36.548054],
                                [827.08057 ,78.347572],
                                [823.72205 ,150.0312 ],
                                [830.97791 ,30.110493],
                                [831.20392 ,64.898506],
                                [832.18945 ,90.032791],
                                [833.87689 ,44.266968],
                                [835.24341 ,94.90918 ],
                                [848.31104 ,68.133461],
                                [849.90454 ,154.76811],
                                [859.0329 ,38.139606],
                                [863.71619 ,36.58371 ],
                                [864.61713 ,70.640457],
                                [865.41138 ,85.65136 ],
                                [868.37506 ,39.971588],
                                [867.74591 ,85.859329],
                                [874.71307 ,48.297085],
                                [881.15955 ,49.452499],
                                [879.73785 ,144.48943],
                                [879.53027 ,149.00845],
                                [901.8584 ,150.29367],
                                [905.28595 ,146.20168],
                                [905.91907 ,152.76945],
                                [936.69983 ,306.85727],
                                [914.28986 ,129.36145],
                                [914.58508 ,141.02995],
                                [937.76453 ,302.45404],
                                [943.66577 ,304.96445],
                                [921.914 ,125.95735],
                                [926.05994 ,151.56125],
                                [950.41528 ,300.37555],
                                [928.94806 ,110.3885 ],
                                [932.60608 ,151.63191],
                                [961.8446 ,298.84912],
                                [942.06323 ,130.65376],
                                [960.1377 ,322.55573],
                                [972.17603 ,298.89777],
                                [950.84448 ,88.339615],
                                [951.04999 ,124.25927],
                                [950.99182 ,129.39116],
                                [950.97864 ,139.78178],
                                [951.97205 ,150.77946],
                                [977.97626 ,298.67361],
                                [954.27472 ,159.0722 ],
                                [954.97534 ,113.22157],
                                [955.0011 ,117.002  ],
                                [954.85986 ,161.78105],
                                [955.97864 ,25.789448],
                                [981.95331 ,305.13092],
                                [956.28113 ,57.53907 ],
                                [957.0545 ,87.909035],
                                [959.72668 ,69.16433 ],
                                [959.99554 ,162.03502],
                                [964.63452 ,118.52385],
                                [965.93127 ,159.21686],
                                [971.3092 ,158.52742],
                                [1002.3577 ,244.32906],
                                [1010.0403 ,266.85059],
                                [1012.8149 ,265.01913]], dtype=np.float32)