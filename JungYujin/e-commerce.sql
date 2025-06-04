CREATE TABLE Customers (
    CustomerID INT AUTO_INCREMENT PRIMARY KEY,
    Gender ENUM('Male', 'Female'),
    MaritalStatus ENUM('Married', 'Single'),
    CityTier INT,
    Tenure FLOAT,
    WarehouseToHome FLOAT,
    HourSpendOnApp FLOAT,
    NumberOfDeviceRegistered INT,
    NumberOfAddress INT
);

CREATE TABLE Devices (
    DeviceID INT AUTO_INCREMENT PRIMARY KEY,
    CustomerID INT,
    PreferredLoginDevice VARCHAR(50),
    NumberOfDeviceRegistered INT,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);

CREATE TABLE Orders (
    OrderID INT AUTO_INCREMENT PRIMARY KEY,
    CustomerID INT,
    PreferedOrderCat VARCHAR(50),
    OrderCount INT,
    OrderAmountHikeFromlastYear FLOAT,
    CouponUsed INT,
    DaySinceLastOrder INT,
    CashbackAmount FLOAT,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);

CREATE TABLE Complaints (
    ComplaintID INT AUTO_INCREMENT PRIMARY KEY,
    CustomerID INT,
    Complain BOOLEAN,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);

CREATE TABLE Satisfaction (
    SatisfactionID INT AUTO_INCREMENT PRIMARY KEY,
    CustomerID INT,
    SatisfactionScore INT,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);

CREATE TABLE Churn (
    ChurnID INT AUTO_INCREMENT PRIMARY KEY,
    CustomerID INT,
    Churn BOOLEAN,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);
