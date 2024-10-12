"use client"

import React, { useState } from "react";
import { Dialog, DialogTitle, DialogContent, DialogActions, Button, Card } from "@mui/material";

interface BuisnessCardProps {
    name: string;
}

export const BuisnessCard = (props: BuisnessCardProps) => {
    const [open, setOpen] = useState(false);

    const handleClickOpen = () => {
        setOpen(true);
    };

    const handleClose = () => {
        setOpen(false);
    };

    return (
        <div style={{width: "100px"}}>
            <Card variant="outlined" onClick={handleClickOpen}>
                <h1 className="font-bold">{props.name}</h1>
            </Card>
            <Dialog open={open} onClose={handleClose}>
                <DialogTitle>Buisness Name</DialogTitle>
                <DialogContent>
                    <p>Buisness Details, perhaps an image?</p>
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleClose}>
                        Exit
                    </Button>
                </DialogActions>
            </Dialog>
        </div>
    );
}