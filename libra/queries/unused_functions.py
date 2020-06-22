# def stats(
#     dataset=None,
#     drop=None,
#     column_name=None,
# ):
#     logger("Reading in dataset....")
#     # Reading in dataset and creating pdtabulate variable to format outputs
#     dataReader = DataReader(dataset)
#     data = dataReader.data_generator()

#     if drop is not None:
#         data.drop(drop, axis=1, inplace=True)

#     data.fillna(0, inplace=True)
#     logger("Creating lambda object to format...")

#     def pdtabulate(df):
#         return tabulate(
#             df, headers='keys', tablefmt='psql')

#     logger("Identifying columns to transform....")

#     # identifying categorical and numerical columns, and encoding
#     # appropriately
#     categor = data.select_dtypes(exclude=['int', 'float'])
#     categor = categor.apply(LabelEncoder().fit_transform)
#     for value in categor.columns:
#         data[str(value)] = categor[str(value)]

#     # if user doesn't specify column analysis on performed on the whole
#     # dataset
#     if column_name == "none":
#         columns = []
#         sim = []
#         for first_val in data.columns:
#             for sec_val in data.columns:
#                 if first_val == sec_val:
#                     continue
#                 columns.append(str(first_val) + "_" + str(sec_val))
#                 sim.append(1 - cosine(data[first_val], data[sec_val]))
#             df = pd.DataFrame(columns=columns)
#             df.loc[len(df)] = sim

#             cols = []
#             vals = []
#             logger("Restructuring dataset for similarity...")
#             # identifying top 5 feature importances and appending them to an
#             # array for display
#             for val in np.argpartition(np.asarray(df.iloc[0]), -5)[-5:]:
#                 cols.append(df.columns[val])
#                 vals.append(df[df.columns[val]].iloc[0])
#                 frame = pd.DataFrame(columns=cols)
#                 frame.loc[len(df)] = vals
#             print("Similarity Spectrum")
#             print(pdtabulate(frame))
#             print()
#             print("Dataset Description")
#             print(pdtabulate(data.describe()))

#         else:
#             logger("Performing similarity calculations....")
#             columns = []
#             sim = []
#             # identifying columns to be compared
#             for val in data.columns:
#                 if val == column_name:
#                     continue
#                 columns.append(str(column_name) + "_" + str(val))
#                 sim.append(1 - cosine(data[column_name], data[val]))
#             df = pd.DataFrame(columns=columns)
#             df.loc[len(df)] = sim

#         cols = []
#         vals = []
#         # identifying top 5 feature importances and appending them to a
#         # dataset
#         for val in np.argpartition(np.asarray(df.iloc[0]), -5)[-5:]:
#             cols.append(df.columns[val])
#             vals.append(df[df.columns[val]].iloc[0])
#             frame = pd.DataFrame(columns=cols)
#             frame.loc[len(df)] = vals

#         # displaying the similarity spectrum and the formatted
#         # data.describe()
#     print("Similarity Spectrum")
#     print("-------------------------")
#     print(pdtabulate(frame))
#     print()
#     print("Dataset Description")
#     print("-------------------------")
#     print(pdtabulate(data[column_name]).describe())


#def booster(instruction,dataset,y,target="",obj):
#    #obj=["reg:linear","multi:softmax "]
#    if target == "":
#        y=data_y(instruction)
#
#    X_train, X_test, y_train, y_test = train_test_split(
#    dataset, y, test_size=0.2, random_state=49)
#    clf = XGBClassifier(objective=obj,learning_rate =0.1,silent=1,alpha = 10)
#    clf.fit(X_train, y_train)
#    y_pred=clf.predict(X_test)
#    return accuracy_score(y_pred, y_test), y_pred 
#    #importance graph
#    #plt.rcParams['figure.figsize'] = [5, 5]
#    #plt.show()