"use client"

import { useState } from 'react';
import { Button, TextField, Typography, Container, Box } from '@mui/material';

export default function BusinessContactForm() {
  const [formData, setFormData] = useState({
    businessName: '',
    contactName: '',
    email: '',
    phone: '',
    address: '',
    description: '',
  });

  const handleChange = (e : any) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = (e : any) => {
    e.preventDefault();
    // Handle form submission logic here
    console.log('Form Data:', formData);
  };

  return (
    <Container maxWidth="sm" className="text-[#365b6d]">
      <Box mt={4}>
        <h1 className="font-bold text-3xl text-center">Let's Get Your Business On Third Space!</h1>
        <br/>
        <p className="text-[#8ca9ad] text-center font-bold">Third Space is the premier platform for connecting your local business with
            consumers! Fill out the form below to get your business listed on our platform before your customers start swiping!</p>
        <form onSubmit={handleSubmit}>
          <TextField
            label="Business Name"
            name="businessName"
            value={formData.businessName}
            onChange={handleChange}
            fullWidth
            margin="normal"
            required
          />
          <TextField
            label="Contact Name"
            name="contactName"
            value={formData.contactName}
            onChange={handleChange}
            fullWidth
            margin="normal"
            required
          />
          <TextField
            label="Email"
            name="email"
            type="email"
            value={formData.email}
            onChange={handleChange}
            fullWidth
            margin="normal"
            required
          />
          <TextField
            label="Phone Number"
            name="phone"
            value={formData.phone}
            onChange={handleChange}
            fullWidth
            margin="normal"
            required
          />
        <TextField
            label="Address"
            name="address"
            value={formData.address}
            onChange={handleChange}
            fullWidth
            margin="normal"
            required
          />
          <TextField
            label="Business Description"
            name="description"
            value={formData.description}
            onChange={handleChange}
            fullWidth
            margin="normal"
            multiline
            rows={4}
            required
          />
          <Box mt={2}>
            <Button type="submit" variant="contained" fullWidth className="bg-[#365b6d]">
              Submit
            </Button>
          </Box>
        </form>
      </Box>
    </Container>
  );
}