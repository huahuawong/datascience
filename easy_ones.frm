/* #1 Write a SQL query for a report that provides the following information for 
      each person in the Person table, regardless if there is an address for each of those people.

      We use left join because we don't care about what's going on the right side, i.e. the address
      If we don't care if there is a last name for each person, we'd use RIHGT JOIN */

    
SELECT Person.FirstName, Person.LastName, Address.City, Address.State 
FROM Person LEFT JOIN Address on Person.PersonId = Address.PersonId;


/* #2 Suppose that a website contains two tables, the Customers table and the Orders table
      Write a SQL query to find all customers who never order anything. */

SELECT Name as Customers from Customers
LEFT JOIN Orders                          # we use LEFT JOIN because we are using the left table, i.e. Customers as reference
ON Customers.Id = Orders.CustomerId
WHERE Orders.CustomerId IS NULL;


