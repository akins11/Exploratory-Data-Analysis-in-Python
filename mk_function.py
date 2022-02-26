import matplotlib.pyplot as plt

# Data cleaning =============================================================================================|
def clean_data(df):
    from pandas import DataFrame, to_datetime, DatetimeIndex
    
    # Removing white-space from the income column name ----------------------------------------------------------------------------------
    df.columns = df.columns.str.replace(" ", "")
    
    # Income ----------------------------------------------------------------------------------------------------------------------------
    df["Income"] = df["Income"].str.replace("$", "", regex=True)
    df["Income"] = df["Income"].str.replace(",", "", regex=True).astype("float")

    # Missing Values
    df["Income"] = df["Income"].fillna(value = df["Income"].median())
    
    # Dt_Customer ----------------------------------------------------------------------------------------------------------------------
    df["Dt_Enroll"] = to_datetime(df["Dt_Customer"])

    # The year the customers joined 
    df["Yr_joined"] = DatetimeIndex(df["Dt_Enroll"]).year

    # The month when customers joined 
    month_nm = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    df["Mth_joined"] = DatetimeIndex(df["Dt_Enroll"]).month

    df["Mth_joined"] =  (
        df["Mth_joined"]
        .replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], month_nm)
        .astype("category")
        .cat.reorder_categories(month_nm, ordered = True)
    )
    
    # Country  ------------------------------------------------------------------------------------------------------------------------
    df["Country"] = df["Country"].replace({"SP": "Spain", "CA": "Canada", "AUS": "Australia", "GER": "Germany",
                                           "IND": "India", "SA" : "South Africa", "ME": "Mexico", "US": "United State"})
    
    # Accepted Campaigns --------------------------------------------------------------------------------------------------------------
    df["Total_Accepted"] = df[[col for col in df.columns if "Accepted" in col]+["Response"]].sum(axis = 1)

    # Changing to string

    for i in [col for col in df.columns if "Accepted" in col][:5]:
        df[i] = df[i].replace({1: "Yes", 0: "No"})
        
    # Amount spent --------------------------------------------------------------------------------------------------------------------
    mnt_col = [col for col in df.columns if "Mnt" in col]

    df["Total_MntSpent"] = df[mnt_col].sum(axis = 1)
    
    # Number of Purchases -------------------------------------------------------------------------------------------------------------
    # I did not include Num of deals Purchased 
    pur_col = [col for col in df.columns if "Purchases" in col][1:]

    df["Total_Purchase"] = df[pur_col].sum(axis = 1) 
    
    # Number of Children and teenagers ------------------------------------------------------------------------------------------------
    df["Dependent"] = df[["Kidhome", "Teenhome"]].sum(axis = 1)
    
    
    # Response and Complain -----------------------------------------------------------------------------------------------------------
    for i in df[["Response", "Complain"]]:
        df[i] = df[i].replace({1: "Yes", 0: "No"})
        
    # Customer's Age ------------------------------------------------------------------------------------------------------------------
    # using 2015 as the recent year
    df["Age"] =  2015 - df["Year_Birth"]
    
    # Marital Status ------------------------------------------------------------------------------------------------------------------
    df["Marital_Status"] = df["Marital_Status"].replace({"YOLO": "Single", "Alone": "Single", "Together": "Married", "Absurd": "Single"})
    
    # Education -----------------------------------------------------------------------------------------------------------------------
    # mrt.groupby("Education")["Age"].describe()

    # based on the median age of customers the education level can be clearly describe as:
    df["Education"] = df["Education"].replace({"Graduation": "Bsc", "2n Cycle": "Undergraduate"})
    
    # Selection and Rearangement of columns --------------------------------------------------------------------------------------------
    df = df[["Income", "Marital_Status", "Kidhome", "Teenhome", 'Dependent', "Dt_Enroll", 'Yr_joined', 'Mth_joined', 'Age', "Education", 'Country',
             "MntWines", 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'Total_MntSpent',                                   
             'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'Total_Purchase', 'NumWebVisitsMonth', "Recency",  
             'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Total_Accepted',                                      
             'Response', 'Complain']]   
    
    return(df)



# Proportion of a variable ==================================================================================|
def prop_table(df, count_var, desc = 2):
    """
    parameters
    ----------
    df:        [DataFrame]
    count_var: [int64, float64] variable to get the proportion of each number.
    desc       [int] number of decimal digits.
    
    return
    ------
    the proportions of the count_var variable.
    """
    from pandas import DataFrame
    
    if (count_var not in df.columns):
        raise Exception(f"argument{count_var} is not in data")
    if (df[count_var].dtype not in ["int64", "float64"]):
        raise Exception(f"argument{count} is not numeric")
    
    prop = round(df[count_var] / df[count_var].sum() * 100, desc)
    return(prop)


# Figuresize ================================================================================================|
def figsize(width_f, height_f):
    """
    width_f: [int] width of the plot.
    height_f [int] height of the plot.
    """
    
#     from matplotlib.pyplot import figure
    
    return plt.figure(figsize = (width_f, height_f))
    
    
    
# labels ====================================================================================================|
def labs(xlabel = None, ylabel = None, title = None, axis_size = None, title_size = None, t_loc = "center"):
    """
    parameters
    ----------
    xlabel: [str] lable for the x axis.
    ylabel: [str] lable for the y axis.
    title:  [str] plot title.
    axis_size: [int] size of the xlabel & ylabel.
    title_size: [int] size of the plot title.
    
    return
    ------
    plt.xlabel, plt.ylabel, plt.title
    """
   
    if axis_size != None:
        lx = plt.xlabel(xlabel, size = axis_size)
        ly = plt.ylabel(ylabel, size = axis_size)
    elif axis_size == None:
        lx = plt.xlabel(xlabel, size = 14)
        ly = plt.ylabel(ylabel, size = 14)
    else:
        None
        
    if title_size != None:
        lt = plt.title(title, size = title_size, loc = t_loc)
    elif title_size == None:
        lt = plt.title(title, size = 17)
    else:
        None
    
    return(lx, ly, lt)


# Annotate on bar plot ========================================================================================|
def bar_text(df, ax, count_var = None, prop_var = None, colr = "black", fs = 13, perc = True, single = False, hue = False, other = False):
    """
    parameters
    ----------
    df        [DataFrame]
    ax        [object] plt.subplot Axes
    count_var [int64] variable containing the count of each obs. in the df.
    prop_var  [float64] variable containing the proportion of each obs. in the df.
    colr      [str] color of the text.
    fs        [int] the size of the text (fontsize).
    perc      [bool] True to add a surffix (%) to the text.
    single    [bool] Annotate on a bar plot without hue.
    hue       [bool] Annotate on a hue bar plot.
    other     [bool] Annotate Another column variable on a bar chart.
    
    returns
    -------
    annotation of text on a barplot.
    """
    # Annotate on a bar plot without hue ----------------------------------------------------------------------|
    if single:
        for index, row in df.iterrows():
            h = row[count_var]
            
            if (h > 0):
                if perc:
                    ax.text(row.name, row[count_var], "{}%".format(row[prop_var]), color = colr, ha = "center", fontsize = fs)
                else:
                    ax.text(row.name, row[count_var], row[prop_var], color = colr, ha = "center", fontsize = fs)
    else:
        None
    # Annotate on a hue bar plot -------------------------------------------------------------------------------|    
    if hue:
        for p in ax.patches:
            h = p.get_height()
            
            if (h > 0):
                if perc:
                    value = "{}%".format(round(100 * p.get_height() / df[count_var].sum(), 2))
                else:
                    value = round(100 * p.get_height() / df[count_var].sum(), 2)
                    
                x = p.get_x() + p.get_width() - 0.09
                y = p.get_height()
                ax.annotate(value, (x, y), ha = "center", color = colr)
    else:
        None
    # Annotate Another column variable on a bar chart ------------------------------------------------------------|    
    if other:
        show = df[prop_var].to_list()
        i = 0
        for p in ax.patches:
            h = p.get_height()
            
            if (h > 0):
                if perc:
                    value = "{}%".format(show[i])
                else:
                    value = show[i]
                    
                x = p.get_x() + p.get_width() - 0.20
                y = p.get_height()
                ax.annotate(value, (x, y), ha = "center", color = colr)
                i = i + 1
    else:
        None
        
        

# Function to create box_plot subplots ================================================================|
def sub_boxplot(df, y_col, axis = None, ylabel = None, title = None):
    """
    df [DataFrame]
    y_col [int64, float64] a numeric variable in the df.
    axis [object] plt.subplot Axes
    ylabel [str] lable for the y axis.
    title [str] plot title.
    """
    from seaborn import boxplot
    
    p = boxplot(data = df, y = y_col, ax = axis)
    p.set_title(title, size = 14)
    p.set(ylabel = ylabel)
    
    return p



# Function to analyse the Year and month Enrollment ======================================================|
def period_exp(df, period_col, col_name, pal = None, title_f = None, tbl = True, plot = True):
    """
    parameter
    ---------
    df         [DataFrame]
    period_col [categorical] year or month variable.
    col_name   [str]
    pal        [str] a list of color for the plot palette.
    tbl        [bool] True to return a summary table.
    plot       [bool] True to return a summary plot.
    
    return
    ------
    plot if plot = True
    table if tbl = True
    table and plot if tbl = True and plot = True
    """
    from pandas import DataFrame
    from seaborn import barplot
    
    # table
    f_tbl = DataFrame(df[period_col].value_counts()).reset_index().rename(columns = {"index": col_name, period_col: "count"}).sort_values(by = col_name)
    f_tbl["prop"] = prop_table(f_tbl, "count", 2)
    f_tbl = f_tbl.reset_index(drop = True)
    
    # plot
    if plot:
        f_figsize = figsize(13, 5)
        p = barplot(data = f_tbl, x = col_name, y = "count", palette = pal)
        
        labs(xlabel = col_name , ylabel = "Number of Customers", 
             title  = f"Count and Proportion Of Customer's Enrollment By {col_name}", 
             axis_size = 15, title_size = 17)
        
        bar_text(df = f_tbl, ax = p, count_var = "count", prop_var = "prop", fs = 14, single = True)
        plt.show()
        
        if tbl:
            return(p, f_tbl)
        else:
            return p
        
    else:
        if tbl:
            return f_tbl
        else:
            raise Exception("any of `tbl` or `plot` must be True")
            
            

# subplots of month by facet year =======================================================================|
def year_month(df, year, axis, xlab_c = False, tick_lab = False):
    """
    parameters
    ---------
    df       [DataFrame]
    year     [int | str] the year to plot.
    axis     [object] plt.subplot Axes.
    xlab_c   [bool] True to add x axis label.
    tick_lab [bool] True to add a axis tick labels.
    
    return
    ------
    facet sub plot of months by year.
    """
    from pandas import DataFrame
    from seaborn import barplot

    f_tbl = period_exp(df = df.query(f"Yr_joined == {str(year)}"), period_col = "Mth_joined", col_name = "month", plot = False)
    f_plt = barplot(data = f_tbl, x = "month", y = "count", palette = "gnuplot", ax = axis)
    f_plt.set_ylabel("Number of Customers", size = 14)
    f_plt.set_title(str(year), size = 17)
    f_plt.xaxis.set_tick_params(labelsize = 15)
    
    if xlab_c:
        f_plt.set_xlabel("Month", size = 14)
    else:
        f_plt.set_xlabel(" ", size = 14)
        
    if tick_lab == False:
        f_plt.xaxis.set_ticklabels([])
    else:
        None
        
    bar_text(df = f_tbl, ax = f_plt, count_var = "count", prop_var = "prop", fs = 13, single = True)
            
    return(f_plt)



# Year, month and country plot ============================================================================================|
def period_country(df, country, axis = None,  title = None, legend_rm = True, title_rm = True, xlabel_in = False):
    """
    parameters
    ----------
    df [DataFrame]
    country     [str] country to plot summary.
    axis        [object] plt.subplot Axes.
    title       [str] plot title.
    legend_rm,  [bool] True to remove legend or title.
    title_rm
    xlabel_in   [bool] True to include x axis label.
    
    return
    ------
    sub plot summary of year month & country
    """
    from pandas import DataFrame
    from seaborn import lineplot
    
    pals = ["crimson", "mediumblue", "aqua"]
    
    p = lineplot(data = df.query(f"Country == '{country}'"), x = "Mth_joined", y = "count", estimator = None, hue = "Yr_joined", palette = pals,  ax = axis)
    p.set_ylabel(country, size = 14)
    
    axis.get_legend().remove() if legend_rm else None
    
    p.set_title("") if title_rm else p.set_title(title, size = 17)
    
    p.set_xlabel("Month Enrolled", size = 15) if xlabel_in else None
    
    return p



# Function for Numerical Distribution
def subplots_n(df, x_col = None, y_col = None, plt_type = None, hue_f = None, pal = None, axis = None, colr = None, bins_f = "auto", kde_f = False, element_f = "bars",
               title_f = None, legend_rm = False, title_rm = True, xlab_in = True, x_lab = None, ylab_in = True, y_lab = None, x_rot = None):
    """
    parameters
    ----------
    df         [DataFrame]
    x_col      [int64 float64], variable from the df to plot.
    y_col      [int64 float64],
    plt_type   [str] The type of plot to create either histogram, boxplot, bar or countplot.
    hue_f      [category, str]
    pal        [str] colors to use for the plot.
    colr       [str] 
    axis       [object] plt.subplot Axes.
    bins_f     [int] the number of bins for the histogram plot.
    kde_f      [bool] True to plot kernel density estimate.
    element_f  [str] for histogram plot, {"bars", "step", "poly"} 
    x_lab,     [str] label for the x axis, y axis and plot title
    y_lab, 
    title_f
    legend_rm, [bool] True to remove legend or title.
    title_rm
    xlab_in,   [bool] True to include x and y axis label
    ylab_in
    x_rot,      [int] rotate x tick labels to . degrees.
    """
    from seaborn import histplot, boxplot, barplot, countplot
    
    # Plot --------------------------------------------------------------------------------------------------------|
    if plt_type == "histogram":
        p = histplot(data = df, x = x_col, y = y_col, hue = hue_f, palette = pal, color= colr, bins= bins_f, kde= kde_f, element= element_f,  ax = axis)
    elif plt_type == "boxplot":
        p = boxplot(data = df, x = x_col, y = y_col, hue = hue_f, palette = pal, ax = axis)
    elif plt_type == "bar":
        p = barplot(data = df, x = x_col, y = y_col, hue = hue_f, palette = pal, ax = axis, errwidth = 1)
    elif plt_type == "countplot":
        p = countplot(data = df, x = x_col, hue = hue_f, palette= pal, ax = axis)
    else:
        raise Exception(f"argument `plt_type` can be any of [histogram, boxplot, bar, countplot] but not {plt_type}")
        
    # Axis labels, title and legend ---------------------------------------------------------------------------------|  
    axis.get_legend().remove() if legend_rm else None
    
    p.set_title("") if title_rm else p.set_title(title_f, size = 17)
    
    p.set_xlabel(x_lab, size = 15) if xlab_in else None
    p.xaxis.set_tick_params(labelrotation = x_rot) if x_rot != None else x_rot
    
    p.set_ylabel(y_lab, size = 15) if ylab_in else None
    
    return p


# Statistical summary of Customer's Age group and Anomunt spent on each Products  =============================================|
def age_stat(df, gp_var, sumy_var, pal = None, drop = False):
    """
    parameters
    ----------
    df       [DataFrame]
    gp_var   [category] a variable from the data[mrt] DataFrame to group by.
    sumy_var [int64, float64] a variable from the data(mrt) DataFrame to summarise.
    pal      [str] a list of color strings for the plot palette.
    drop     [bool] True to use only [median, max, sum] to summarise the data else [min, median, max, sum].
    
    return
    ------
    categorical plot of statistical description bars.
    """
    
    from pandas import DataFrame, melt
    from seaborn import catplot
    
    f_tbl = df.groupby(gp_var)[sumy_var].agg(["min", "median", "std", "max", "sum"]).reset_index()
    f_tbl = f_tbl.rename(columns = {"min": "Minimun", "median": "Median", "std": "STD", "max": "Maximum", "sum": "Sum"})
    
    # --------------------------------------------------------------------------------------------------------------------------
    if drop:
        ff_tbl = melt(f_tbl, id_vars = [gp_var], value_vars = ["Median", "Maximum", "Sum"])
        g_f = catplot(data = ff_tbl, x = gp_var, y = "value", col = "variable", kind = "bar", sharey = False, palette= pal)
    
        axes = g_f.axes.flatten()
        axes[0].set_title("Median", size = 16)
        axes[1].set_title("Maximum Amount", size = 16)
        axes[2].set_title("Total Sum", size = 16)
    else:
        ff_tbl = melt(f_tbl, id_vars = [gp_var], value_vars = ["Minimun", "Median", "Maximum", "Sum"])
        g_f = catplot(data = ff_tbl, x = gp_var, y = "value", col = "variable", kind = "bar", sharey = False, palette= pal)

        axes = g_f.axes.flatten()
        axes[0].set_title("Minimun Amount", size = 16)
        axes[1].set_title("Median", size = 16)
        axes[2].set_title("Maximum Amount", size = 16)
        axes[3].set_title("Total Sum", size = 16)

    g_f.set_axis_labels(x_var = "Customer's Age Group", y_var = "Value", size = 14)
    # ---------------------------------------------------------------------------------------------------------------------------
    return(f_tbl)





# Customer's age group and product purchased summary  ===================================================================|
def age_purchase(df, gp_var, sumy_var, pal_f = None):
    """
    parameter
    ---------
    df       [DataFrame]
    gp_var   [category] a variable from the df DataFrame to group by.
    sumy_var [int64, float64] a variable from the df DataFrame to summarise.
    lab      [str] plot title.
    pal_f    [str] a list of color strings for the plot palette.
    
    return
    -------
    A summary boxplot and bar chart.
    """
    from pandas import DataFrame
    from regex import sub
    
    f_tbl = df.groupby(gp_var)[sumy_var].sum().reset_index(name = "Sum")
    # -----------------------------------------------------------------------------------------------
#     title = "Total Number Of Customer" if lab == "Total" else f"Customer {lab}"
    
    til = sub(r"([A-Z])", r" \1", sumy_var).replace("Num", "").replace("Purchases", "").strip()
    title = "Total Number Of Customer" if sumy_var == "Total_Purchase" else f"Customer {til}"
    lab = til if sumy_var != "Total_Purchase" else "Total"
    
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (17, 5))
    
    subplots_n(df = df, x_col = gp_var, y_col = sumy_var, axis = ax1, plt_type = "boxplot", 
               x_lab = "Customer's Age Group", y_lab = f"Number Of {lab} Purchases", title_f = f"Age Group and {title} Purchases",  
               xlab_in = True, ylab_in = True, title_rm = False, pal = pal_f)
    
    subplots_n(df = f_tbl, x_col = gp_var, y_col = "Sum", axis = ax2, plt_type = "bar", 
               x_lab = "Customer's Age Group", y_lab = f"Sum Of {lab} Purchases", xlab_in = True, ylab_in = True, pal = pal_f)
    
    
    
    
# The Summary of Customers Age and Campaign performance =======================================================================
def age_campaign(df, gp_var, pal = None, response = False, hue_ord= None):
    """
    parameter
    ---------
    df       [DataFrame]
    gp_var   [category] a variable from the df DataFrame to group by.
    pal      [str] a list of color strings for the plot palette.
    response [bool] to include a response variable title.
    title_f  [str] plot name for the title.
    hue_ord  [str] Order to plot the categorical levels.
    
    returns
    -------
    plot summary of accepted campaign by customers age category.
    """
    from pandas import DataFrame
    from seaborn import barplot
    from numpy import nan
    
    f_tbl = df.groupby([gp_var, "Age_cat"])[gp_var].value_counts()
    f_tbl.index = f_tbl.index.droplevel(0)
    f_tbl = f_tbl.reset_index(name = "count")
    # --------------------------------------------------------------+
    f_tbl["prop_by_age"] = nan
    
    for i in f_tbl["Age_cat"].unique():
        f_tbl.loc[f_tbl["Age_cat"] == i, "prop_by_age"] = round(f_tbl.loc[f_tbl["Age_cat"] == i]["count"] / f_tbl.loc[f_tbl["Age_cat"] == i]["count"].sum() * 100, 1)
    # --------------------------------------------------------------+
    no = " ".join([i for i in gp_var if i.isdigit()])
    camp_til = {"1": "First", "2": "Second", "3": "Third", "4": "Fourth", "5": "Fifth"}
    
    figsize(14, 5)
    pp_f = barplot(data = f_tbl, x = "Age_cat", y = "count", hue = gp_var, palette= pal, linewidth = 4, hue_order= hue_ord)
    bar_text(df = f_tbl, ax = pp_f, prop_var = "prop_by_age", other = True)
    
    if response:
        labs(xlabel = "Customer Age Group", ylabel= "Count", title = "Number & Proportion of Customer's Response to the last Campaign by Age", 
             axis_size = 15, title_size = 17)
    else:
        title_f = camp_til[no] if gp_var != "Complain" else ""
        labs(xlabel = "Customer Age Group", ylabel = "Count", title = f"Number & Proportion of Customers that Accepted the {str(title_f)} Campaign by Age", 
             axis_size= 15, title_size= 17)
        
        
        
        
# income correlations with other variables ==========================================================        
def income_corr(df, only_df = False, variables = None, only_iv = True):
    """
    parameters
    ----------
    df        [DataFrame]
    only_df   [bool] only use the df data to compute the correlation.
    variables [int64, float64] a list of variables from the df data.
    only_iv   [bool] correlation for only income and the inputed variables.
    
    returns
    -------
    a table containing the correlation between income in the df and the inputed variables.
    """
    from pandas import DataFrame, concat
    
    if only_iv:
        if only_df:
            output = (df.corr(method = 'kendall')["Income"].sort_values(ascending = False)
                      .reset_index(name = "correlation").rename(columns = {'index':"variable"}) )
        else:
            output = (df[variables]
                      .corr()["Income"].sort_values(ascending = False)
                      .reset_index(name = "correlation").rename(columns = {'index':"variable"}) )
        return output

    else:
        def f_corr(response, c_df = df, var = variables):
            corr_name = {"Yes": "Responded", "No": "No Response"}
            index_nm = corr_name[response]

            output = (c_df.query(f"Response == '{response}'")[var]
                      .corr()["Income"]
                      .sort_values(ascending = False)
                      .reset_index(name = index_nm)
                      .rename(columns = {'index':"variable"}) )
            return output

        corr_yes = f_corr(response = "Yes")
        corr_no  = f_corr(response = "No")

        return concat([corr_yes, corr_no["No Response"]], axis = 1)
    
    
    

# variable summary ==============================================================================
def var_sumy(df, var, fun = "sum", pals = None, x_lab = None, y_lab = None, title_f = None):
    """
    parameters
    ----------
    df    [DataFrame]
    var   [int64, float64]  a list of variables from the df data.
    fun   [str] the type of function to use for the summary any of [sum, median, mean].
    pals  [str] a list of color strings for the plot palette.
    xlab, [str] lable for the x and y axis and the plot title.
    ylab,
    title_f
    
    return
    ------
    plot summary for all inputed variables from the df data.
    """
    from pandas import DataFrame
    from seaborn import barplot
    
    if fun == "sum":
        f_tbl = df[var].sum()
    elif fun == "mean":
        f_tbl = df[var].mean()
    elif fun == "median":
        f_tbl = df[var].median()
    else:
        raise Exception(f"argument `fun` can be any of [sum, mean, median] but not {fun}")
        
    f_tbl = f_tbl.sort_values(ascending = False).reset_index(name = "col_2").rename(columns = {"index": "col_1"})
    f_tbl["prop"] = prop_table(f_tbl, "col_2", 2)
    # --------------------------------------------------------------------------------------------------------------------
    
    f_plt = barplot(data = f_tbl, x = "col_1", y = "col_2", palette = pals)
    
    labs(xlabel= x_lab, ylabel = y_lab, title = title_f, axis_size = 15, title_size = 17)
    bar_text(df = f_tbl, ax = f_plt, count_var = "col_2", prop_var = "prop", single = True)
    
    
    
# Grouped product summary ====================================================================================================|
def products_vars(df, gp_var, pals = None, rot = None):
    """
    parameter
    ---------
    df     [DataFrame]
    gp_var [str, object] a variable from the df DataFrame to group by.
    pals   [str] a list of color strings for the plot palette.
    
    return
    ------
    plot summary of the total amount of the various products by the grouped variable `gp_var`.
    """
    
    from pandas import DataFrame
    
    amount_lst = ["MntWines", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntFruits", "Total_MntSpent"]
    
    f_df = (df.groupby(gp_var)[amount_lst]
          .sum()
          .reset_index()
          .sort_values(by = "Total_MntSpent", ascending = False))
    
    f_df["count"] = df[gp_var].value_counts().values
    
    # plot ---------------------------------------------------------------------------------------------------------------------------------------------
    title = str.title(str.replace(gp_var, "_", " "))
    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (20, 10))
    
    subplots_n(df = df, x_col= gp_var, y_col= amount_lst[5], axis= axs[0,0], hue_f = "Response", plt_type= 'bar', y_lab= "Total Amount Spent",
               legend_rm= True, pal= pals, title_f= f"Average amount Spent On Each Products with {str(title)} and Campaign Response", title_rm= False, x_rot = rot)
    
    subplots_n(df= df, x_col= gp_var, y_col= amount_lst[0], axis= axs[0,1], hue_f ="Response", pal= pals, plt_type= 'bar', y_lab= "Wine", legend_rm= True, x_rot = rot)
    subplots_n(df= df, x_col= gp_var, y_col= amount_lst[1], axis= axs[0,2], hue_f ="Response", pal= pals, plt_type= 'bar', y_lab= "Meat", x_rot = rot)
    subplots_n(df= df, x_col= gp_var, y_col= amount_lst[2], axis= axs[1,0], hue_f ="Response", pal= pals, plt_type= 'bar', y_lab= "Fish", legend_rm= True, x_rot = rot)
    subplots_n(df= df, x_col= gp_var, y_col= amount_lst[3], axis= axs[1,1], hue_f ="Response", pal= pals, plt_type= 'bar', y_lab= "Sweet", legend_rm= True, x_rot = rot)
    subplots_n(df= df, x_col= gp_var, y_col= amount_lst[4], axis= axs[1,2], hue_f ="Response", pal= pals, plt_type= 'bar', y_lab= "Fruits", legend_rm= True, x_rot = rot)
    return(f_df)


# summary of purchase by country
def purchase_country(df, agg_var):
    """
    parameter
    ---------
    df      [DataFrame]
    agg_var [int64, float64] a purchase variable from the data df. 
    
    returns
    -------
    plot summary of the purchase variable `agg_var` by country.
    """
    
    from regex import sub
    from pandas import DataFrame
    
    f_tbl = df.groupby("Country")[agg_var].agg(["mean", "sum"]).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (15, 5))
    
    til = "Total Purchase" if agg_var == "Total_Purchase" else sub(r"([A-Z])", r" \1", agg_var).replace("Num", "").strip()
        
    pal_f = {"Spain": "indigo", "South Africa": "darkorange", "Canada": "cyan", "Australia": "crimson", "India": "forestgreen", 
             "Germany": "royalblue", "United State": "sienna", "Mexico": "teal"}
    
    subplots_n(df = f_tbl.sort_values(by = "mean", ascending = False), 
               y_col= "Country", x_col= "mean", axis= ax1, plt_type= 'bar', y_lab= "Country", x_lab= "Average Purchase",
               title_f= f"{str(til)} With Country", xlab_in= True, ylab_in= True, title_rm= False, pal = pal_f)
    
    subplots_n(df = f_tbl.sort_values(by = "sum", ascending = False), 
               y_col= "Country", x_col= "sum", axis= ax2, plt_type= 'bar', x_lab= "Sum Of Purchase", xlab_in= True,
              pal = pal_f)   
    
    

# Campaign Count ========================================================================================================|
def campaign_count(df, campaign, axi, x_labc = "", y_labc = "", palc = None):
    """
    parameter
    ----------
    df       [DataFrame]
    campaign [int64, object] The type of campaign.
    axi      [object] plt.subplot Axes.
    x_labc,  [str] label for the x and y axis.
    y_labc
    palc     [str] a list of color strings for the plot palette.
    
    return
    ------
    summary plot of campaign acceptance count.
    """
    
    from pandas import DataFrame
    
    f_tbl = df[campaign].value_counts().reset_index(name = "count")
    f_tbl["prop"] = prop_table(f_tbl, "count")
    
    no = " ".join([i for i in campaign if i.isdigit()])
    til = f"Campaign {no}" if campaign != "Total_Accepted" else "Total Number Of Campaign Accepted"
    
    f_plt = subplots_n(df = f_tbl, x_col = "index", y_col = "count", axis = axi, 
                       y_lab = y_labc, ylab_in = True, x_lab = x_labc, xlab_in = True, title_f = til, title_rm = False, 
                       plt_type = "bar", pal = palc)
    
    bar_text(df = f_tbl, ax = f_plt, count_var = "count", prop_var = "prop", single= True)
    

    
# influence of campaign on each products =====================================================================================|
def campaign_product(df, campaign):
    """
    df [DataFrame]
    campaign [category object] any campaign variable in the df to sort by.
    
    return
    ------
    summary plot of accepted campaigns by amount spent on all products.
    """
    
    from pandas import DataFrame
    from seaborn import barplot
    
    f_tbl = (df.query(f"{str(campaign)} == 'Yes'")[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']]
             .agg(["mean", "sum"])
             .T
             .reset_index()
             .sort_values(by = "mean", ascending= False))
    
    figsize(14, 5)
    
    no = " ".join([i for i in campaign if i.isdigit()])
    camp_til = {"1": "First", "2": "Second", "3": "Third", "4": "Fourth", "5": "Fifth"}
    til = camp_til[no]
    
    f_plt = barplot(data = f_tbl, x = "sum", y = "index", orient = "h", palette = "brg")
    labs(xlabel = "Sum of Amount Spent", ylabel = "Product and (Average Amount)", title = f"{til} Campaign With Amount Spent On Each Product")
    
    show = round(f_tbl["mean"], 2).to_list()
    i = 0
    for p in f_plt.patches:
        plt.text(p.get_width(), p.get_y()+0.55*p.get_height(), show[i], va = "center")
        i = i+1 
        
    pr = f_tbl["sum"].sum()
    
    print(f"<|Total Amount Spent on all Products for customers who Accepted the {til} Campaign is {pr:,}|>", "\n")